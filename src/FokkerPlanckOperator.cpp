#define MFEM_USE_MPI

#include "FokkerPlanckOperator.hpp"

using namespace mfem;
using namespace fokker;

FokkerPlanckOperator::FokkerPlanckOperator(ParFiniteElementSpace &fes,
										   const Array<int> ess_tdofs,
										   VectorCoefficient &space_adv_coeff,
										   MatrixArrayCoefficient &conf_adv_coeff,
										   const FokkerPlanckSettings &settings)
	: FPOperator(settings.GetNModes(fes.GetMesh()->Dimension()) * fes.GetTrueVSize()), 
	  fespace(fes),
	  ess_tdof_list(ess_tdofs),
	  settings(settings),
	  Mform(&fes),
	  Kform(&fes),
	  Cform(fes.GetMesh()->Dimension() * fes.GetMesh()->Dimension()),
	  Cmat(fes.GetMesh()->Dimension() * fes.GetMesh()->Dimension()),
	  Tmat(1 + settings.GetNPairs()),
	  M_solver(fes.GetComm()),
	  T_solver(fes.GetComm()),
	  block_offsets({0, fes.GetTrueVSize(), 2 * fes.GetTrueVSize()}),
	  fp_op(block_offsets),
	  fp_op_solver(fes.GetComm()),
	  fp_op_prec(block_offsets),
	  tmp(fes.GetTrueVSize()),
	  btmp(block_offsets),
	  btmp2(block_offsets)
{
	MFEM_ASSERT(fes.GetMesh()->Dimension() == 2,
				"Fokker-Planck-Operator for dimension =! 2 is not implemented.");

	// Mass matrix
	Mform.AddDomainIntegrator(new MassIntegrator);
	Mform.Assemble(0); // keep sparsity pattern of all matrices the same
	Mform.FormSystemMatrix(ess_tdof_list, Mmat);

	// Diffusion matrix
	ConstantCoefficient diff_coeff(-settings.space_diff_coeff); // Diffusion has opposite sign than usual
	Kform.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
	Kform.Assemble(0); // keep sparsity pattern of all matrices the same
	Kform.FormSystemMatrix(ess_tdof_list, Kmat);
	Array<int> empty;
	Kform.FormSystemMatrix(empty, Kmat0); // no BC

	Cform = nullptr;
	for (auto &C : Cmat)
	{
		C = new HypreParMatrix;
	}

	// Solvers for mass and implicit operator
	M_solver.iterative_mode = false;
	M_solver.SetRelTol(settings.rel_tol);
	M_solver.SetAbsTol(settings.abs_tol);
	M_solver.SetMaxIter(settings.max_iter);
	M_solver.SetPrintLevel(settings.print_level_M);
	M_prec.SetType(HypreSmoother::Jacobi);
	M_solver.SetPreconditioner(M_prec);
	M_solver.SetOperator(Mmat);

	fp_op_solver.iterative_mode = false;
	fp_op_solver.SetRelTol(settings.rel_tol);
	fp_op_solver.SetAbsTol(settings.abs_tol);
	fp_op_solver.SetMaxIter(settings.max_iter);
	fp_op_solver.SetPrintLevel(settings.print_level_fp);
	fp_op_solver.SetPreconditioner(fp_op_prec);

	T_solver.iterative_mode = false;
	T_solver.SetRelTol(1e-6);
	T_solver.SetAbsTol(1e-100);
	T_solver.SetMaxIter(100);
	T_solver.SetPrintLevel(settings.print_level_T);
	T_solver.SetPreconditioner(M_solver);
	Tmat = nullptr;

	// initialize parameter function
	SetParameters(space_adv_coeff, conf_adv_coeff);
}

void FokkerPlanckOperator::Mult(const Vector &phi, Vector &dphi_dt) const
{
	const int vsize = fespace.TrueVSize();
	{ // Solve 0th mode
		const Vector phi_0(phi.GetData(), vsize);
		Vector dphi_0_dt(dphi_dt.GetData(), vsize);
		// apply the advection-diffusion solver operator
		Amat.Mult(phi_0, tmp);
		Kmat.AddMult(phi_0, tmp);
		M_solver.Mult(tmp, dphi_0_dt);
		dphi_0_dt.SetSubVector(ess_tdof_list, 0.0);

		MFEM_VERIFY(M_solver.GetConverged(), "Mass solver did not converge.");
		if (settings.verbose)
		{
			out << std::setw(8) << ""
				<< std::setw(5) << "Iter "
				<< std::setw(8) << "Residual" << std::endl;
			out << std::setw(8) << "phi_0   "
				<< std::setw(4) << M_solver.GetNumIterations() << " "
				<< std::setw(8) << std::setprecision(2) << std::scientific
				<< M_solver.GetFinalNorm() << std::defaultfloat << std::endl;
		}
	}
	// Solve even and odd modes
	for (int k = 1; k <= settings.GetNPairs(); ++k)
	{
		const int mode = settings.IndexToMode(k);
		const int lower_index = std::max(2 * settings.ModeToIndex(mode - 2) - 1, 0);
		const BlockVector phi_k(phi.GetData() + (2 * k - 1) * vsize, block_offsets);
		const BlockVector phi_km1(phi.GetData() + lower_index * vsize, block_offsets);
		BlockVector dphi_k_dt(dphi_dt.GetData() + (2 * k - 1) * vsize, block_offsets);
		ApplyForwardMode(phi_k, phi_km1, mode, btmp);
		for (int i = 0; i < 2; ++i)
		{
			M_solver.Mult(btmp.GetBlock(i), dphi_k_dt.GetBlock(i)); // apply M^{-1}
			dphi_k_dt.GetBlock(i).SetSubVector(ess_tdof_list, 0.0); // correct? remove essential BCs

			MFEM_VERIFY(M_solver.GetConverged(), "Mass solver did not converge.");
			if (settings.verbose)
			{
				std::string name = "phi^";
				name += (i ? "s_" : "c_");
				name += std::to_string(mode) + " ";
				out << std::setw(8) << name
					<< std::setw(4) << M_solver.GetNumIterations() << " "
					<< std::setw(8) << std::setprecision(2) << std::scientific
					<< M_solver.GetFinalNorm() << std::defaultfloat << std::endl;
			}
		}
	}
}

void FokkerPlanckOperator::ImplicitSolve(const double dt,
										 const Vector &phi,
										 Vector &dphi_dt)
{
	// TODO: Currently only applies a block-diagonal preconditioner to solve
	//	 |M - dt (A+K-R_m)   - dt C_m      | |dphi_{c,m}| = |RHS(phi_{c,m}, phi_{c,m-2} + dt dphi_{c,m-2})|
	//	 |    dt C_m       M - dt (A+K-R_m)| |dphi_{s,m}| = |RHS(phi_{s,m}, phi_{c,m-2} + dt dphi_{s,m-2})|
	if (!Tmat[0] || dt != current_dt)
	{
		// Set diagonal operator T_m = M - dt * (A-K-R_m) for all modes
		for (int k = 0; k <= settings.GetNPairs(); ++k)
		{
			delete Tmat[k];
			Tmat[k] = Add(1.0 + dt * GetReactionCoeff(settings.IndexToMode(k)), Mmat, -dt, Amat);
			Tmat[k]->Add(-dt, Kmat);
		}
		current_dt = dt;
	}

	const int vsize = fespace.TrueVSize();
	{ // Solve 0th mode
		const Vector phi_0(phi.GetData(), vsize);
		Vector dphi_0_dt(dphi_dt.GetData(), vsize);
		Amat0.Mult(phi_0, tmp); // TODO: correct? If so, also necessary in ApplyForwardMode
		Kmat0.AddMult(phi_0, tmp);
		T_solver.SetOperator(*Tmat[0]);
		T_solver.Mult(tmp, dphi_0_dt);
		dphi_0_dt.SetSubVector(ess_tdof_list, 0.0); // TODO: correct?

		MFEM_VERIFY(T_solver.GetConverged(), "Implicit solver did not converge.");
		if (settings.verbose)
		{
			out << std::setw(8) << ""
				<< std::setw(5) << "Iter "
				<< std::setw(8) << "Residual" << std::endl;
			out << std::setw(8) << "phi_0   "
				<< std::setw(4) << T_solver.GetNumIterations() << " "
				<< std::setw(8) << std::setprecision(2) << std::scientific
				<< T_solver.GetFinalNorm() << std::defaultfloat << std::endl;
		}
	}
	// Solve even and odd modes
	for (int k = 1; k <= settings.GetNPairs(); ++k)
	{
		const int mode = settings.IndexToMode(k);
		const int lower_index = std::max(2 * settings.ModeToIndex(mode - 2) - 1, 0);
		const BlockVector phi_k(phi.GetData() + (2 * k - 1) * vsize, block_offsets);
		const BlockVector phi_km1(phi.GetData() + lower_index * vsize, block_offsets);
		const BlockVector dphi_km1_dt(dphi_dt.GetData() + lower_index * vsize, block_offsets);
		BlockVector dphi_k_dt(dphi_dt.GetData() + (2 * k - 1) * vsize, block_offsets);
		add(phi_km1, dt, dphi_km1_dt, btmp2);		// btmp = phi_m-2 + dt dphi_m-2
		ApplyForwardMode(phi_k, btmp2, mode, btmp); // TODO: correct?

		// diagonal preconditioner
		T_solver.SetOperator(*Tmat[k]);
		fp_op_prec.SetDiagonalBlock(0, &T_solver);
		fp_op_prec.SetDiagonalBlock(1, &T_solver);
		// block operator
		fp_op.SetDiagonalBlock(0, Tmat[k]);
		fp_op.SetDiagonalBlock(1, Tmat[k]);
		// TODO should Cmat[3] +- Cmat[2] instead for k==1
		fp_op.SetBlock(0, 1, Cmat[3], -dt * mode);
		fp_op.SetBlock(1, 0, Cmat[3], dt * mode);
		fp_op_solver.SetOperator(fp_op);
		fp_op_solver.Mult(btmp, dphi_k_dt);
		for (int i = 0; i < 2; ++i)
		{
			dphi_k_dt.GetBlock(i).SetSubVector(ess_tdof_list, 0.0); // TODO: correct?
		}
		MFEM_VERIFY(fp_op_solver.GetConverged(), "Implicit solver did not converge.");

		if (settings.verbose)
		{
			std::string name = "phi_";
			name += std::to_string(mode) + "   ";
			out << std::setw(8) << name
				<< std::setw(4) << fp_op_solver.GetNumIterations() << " "
				<< std::setw(8) << std::setprecision(2) << std::scientific
				<< fp_op_solver.GetFinalNorm() << std::defaultfloat << std::endl;
		}
	}
}

void FokkerPlanckOperator::SetParameters(VectorCoefficient &space_adv_coeff,
										 MatrixArrayCoefficient &conf_adv_coeff)
{
	// Update spatial advection form
	delete Aform;
	Aform = new ParBilinearForm(&fespace);
	Aform->AddDomainIntegrator(new ConvectionIntegrator(space_adv_coeff, -1.0));
	Aform->Assemble(0); // keep sparsity pattern of all matrices the same
	Aform->FormSystemMatrix(ess_tdof_list, Amat);
	Array<int> empty;
	Aform->FormSystemMatrix(empty, Amat0); // no BC

	// Update coupling forms
	// TODO: extend to 3D
	for (int i = 0; i < 4; ++i)
	{
		delete Cform[i];
		Cform[i] = new ParBilinearForm(&fespace);
		SumCoefficient coeff(*conf_adv_coeff.GetCoeff(0, i / 2), *conf_adv_coeff.GetCoeff(1, 1 - i / 2), 0.5, (i % 2) ? -0.5 : 0.5);
		Cform[i]->AddDomainIntegrator(new MassIntegrator(coeff));
		Cform[i]->Assemble(0);								 // keep sparsity pattern of all matrices the same
		Cform[i]->FormSystemMatrix(ess_tdof_list, *Cmat[i]); // TODO: correct?
	}

	// Delete implicit time-stepping operators for rebuild
	for (auto *T : Tmat)
	{
		delete T;
	}
	Tmat = nullptr;
}

void FokkerPlanckOperator::ApplyAKRm(const Vector &x, const double r, Vector &y) const
{
	Amat.Mult(x, y);
	Kmat.AddMult(x, y);
	Mmat.AddMult(x, y, -r);
}

void FokkerPlanckOperator::ApplyForwardMode(const BlockVector &phi_m,
											const BlockVector &phi_mm2,
											const int m,
											BlockVector &y) const
{
	auto &phi_c = phi_m.GetBlock(0);
	auto &phi_s = phi_m.GetBlock(1);
	auto &phi_m2c = phi_mm2.GetBlock(0);
	auto &phi_m2s = phi_mm2.GetBlock(1);
	auto &y_c = y.GetBlock(0);
	auto &y_s = y.GetBlock(1);

	const double r_m = GetReactionCoeff(m);

	const double coeff = FokkerPlanckCoefficients::Coefficient(m, settings.potential, fespace.GetMesh()->Dimension());

	// cosine part
	ApplyAKRm(phi_c, r_m, y_c);		 // advection-diffusion-reaction part
	Cmat[0]->AddMult(phi_c, y_c, m); // trace reaction part
	Cmat[3]->AddMult(phi_s, y_c, m); // asymetric coupling part
	if (m == 1)
	{
		Cmat[1]->AddMult(phi_c, y_c); // extra reaction part
		Cmat[2]->AddMult(phi_s, y_c); // extra symmetric coupling part
	}
	else // m > 1, coupling down to phi_{m-2}
	{
		Cmat[1]->AddMult(phi_m2c, y_c, (m == 2) ? 2 * coeff : coeff);
		if (m > 2)
		{
			Cmat[2]->AddMult(phi_m2s, y_c, -coeff);
		}
	}

	// sine part
	ApplyAKRm(phi_s, r_m, y_s);		  // advection-diffusion-reaction part
	Cmat[0]->AddMult(phi_s, y_s, m);  // trace reaction part
	Cmat[3]->AddMult(phi_c, y_s, -m); // asymetric coupling part
	if (m == 1)
	{
		Cmat[1]->AddMult(phi_s, y_s, -1.0); // extra reaction part
		Cmat[2]->AddMult(phi_c, y_s);		// extra symmetric coupling part
	}
	else // m > 1, coupling down to phi_{m-2}
	{
		Cmat[2]->AddMult(phi_m2c, y_s, (m == 2) ? 2 * coeff : coeff);
		if (m > 2)
		{
			Cmat[1]->AddMult(phi_m2s, y_s, coeff);
		}
	}
}

double FokkerPlanckOperator::GetReactionCoeff(const int mode) const
{
	return settings.conf_diff_coeff * 2 * mode * FokkerPlanckCoefficients::Coefficient(mode, settings.potential, fespace.GetMesh()->Dimension());
}

FokkerPlanckOperator::~FokkerPlanckOperator()
{
	delete Aform;
	for (auto *C : Cform)
	{
		delete C;
	}
	for (auto *C : Cmat)
	{
		delete C;
	}
	for (auto *T : Tmat)
	{
		delete T;
	}
}