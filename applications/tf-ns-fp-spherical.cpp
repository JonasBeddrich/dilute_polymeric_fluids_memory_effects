// This program solves the time-fractional Navier-Stokes-Fokker-Planck system
// using an spherical harmonics ansatz in the configuration space leading to a
// lower block-triangular system of advection-diffusion-reaction PDEs for the
// Fokker-Planck equation.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "Coefficient.hpp"
#include "FokkerPlanckSolver.hpp"
#include "Setups.hpp"

using namespace mfem;
using namespace fokker;
using namespace navier;

void vel(const Vector &x, double t, Vector &u)
{
	double xi = x(0);
	double yi = x(1);

	u(0) = M_PI * sin(t) * pow(sin(M_PI * xi), 2.0) * sin(2.0 * M_PI * yi);
	u(1) = -(M_PI * sin(t) * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0));
}

double p(const Vector &x, double t)
{
	return cos(M_PI * x(0)) * sin(t) * sin(M_PI * x(1));
}

constexpr double kin_viscosity = 1.0;
void accel(const Vector &x, double t, Vector &u)
{
	double xi = x(0);
	double yi = x(1);

	u(0) = M_PI * sin(t) * sin(M_PI * xi) * sin(M_PI * yi) * (-1.0 + 2.0 * pow(M_PI, 2.0) * sin(t) * sin(M_PI * xi) * sin(2.0 * M_PI * xi) * sin(M_PI * yi)) + M_PI * (2.0 * kin_viscosity * pow(M_PI, 2.0) * (1.0 - 2.0 * cos(2.0 * M_PI * xi)) * sin(t) + cos(t) * pow(sin(M_PI * xi), 2.0)) * sin(2.0 * M_PI * yi);

	u(1) = M_PI * cos(M_PI * yi) * sin(t) * (cos(M_PI * xi) + 2.0 * kin_viscosity * pow(M_PI, 2.0) * cos(M_PI * yi) * sin(2.0 * M_PI * xi)) - M_PI * (cos(t) + 6.0 * kin_viscosity * pow(M_PI, 2.0) * sin(t)) * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0) + 4.0 * pow(M_PI, 3.0) * cos(M_PI * yi) * pow(sin(t), 2.0) * pow(sin(M_PI * xi), 2.0) * pow(sin(M_PI * yi), 3.0);
}

int main(int argc, char *argv[])
{
	Mpi::Init();
	Hypre::Init();

	StopWatch sw_all;
	sw_all.Clear();
	sw_all.Start();

	////////////////////////////////////////////////////////////////////////////
	// Define and read options
	NSFP_RunSetup setup;
	int output_steps = 1;
	double scale_T = 1.0; // ???
	double alpha = 1.0;	  // parameter of the fractional temporal derivative
	int N_kernel = 1;	  // number of kernel-compression modes

	err << "WARNING: NO TIME-FRACTIONAL SOLVER IMPLEMENTED!" << std::endl
		<< "\tusing standard ODE instead ..." << std::endl
		<< std::endl;

	OptionsParser args(argc, argv);
	setup.setOptions(args);
	args.AddOption(&output_steps, "-os", "--output-steps", "Output solution every n-th timestep.");
	args.ParseCheck();

	////////////////////////////////////////////////////////////////////////////
	// initialize the runcase
	setup.initialize();
	// setup.fp_settings.print_level_fp = setup.fp_settings.print_level_fp.All();
	// setup.fp_settings.print_level_T = setup.fp_settings.print_level_T.All();
	ParMesh &pmesh = setup.GetMesh();

	H1_FECollection fec(setup.fp_settings.order, pmesh.Dimension());
	ParFiniteElementSpace fes(&pmesh, &fec);
	ParFiniteElementSpace vfes(&pmesh, &fec, pmesh.Dimension());

	if (Mpi::Root())
	{
		out << std::endl
			<< "Number of elements: " << pmesh.GetNE() << std::endl
			<< std::endl;
	}

	////////////////////////////////////////////////////////////////////////////
	// Define the boundary conditions and parameter coefficients

	Array<int> dbc_bdr(pmesh.bdr_attributes.Max());	   // Dirichlet BCs
	Array<int> outflow_bc(pmesh.bdr_attributes.Max()); // Outflow BCs
	dbc_bdr = 0;
	if (pmesh.bdr_attributes.Max() > 0)
	{
		// dbc_bdr[1] = 1; no Dirichlet BC for now
	}
	outflow_bc = 0;
	if (pmesh.bdr_attributes.Max() > 1)
	{
		// outflow_bc[2] = 1; // no Outflow BC for now
	}

	Array<int> domain_attr(pmesh.attributes.Max());
	domain_attr = 1;

	////////////////////////////////////////////////////////////////////////////
	// Prepare ParaView output
	std::string pathname("NSFP_" + std::to_string(setup.fp_settings.compute_odd_modes));
	pathname += "_K" + std::to_string(setup.fp_settings.max_mode);
	pathname += "_dt" + std::to_string(setup.time_settings.dt);
	ParaViewDataCollection dataCollection(pathname, &pmesh);
	dataCollection.SetPrefixPath("ParaView/NSFP_Case" + std::to_string(setup.run_case));
	dataCollection.SetLevelsOfDetail(setup.fp_settings.order);
	dataCollection.SetDataFormat(VTKFormat::BINARY);
	dataCollection.SetHighOrderOutput(true);

	////////////////////////////////////////////////////////////////////////////
	// Define the PDE/ODE solvers and add to ParaView output
	FokkerPlanckSolver fp_solver(pmesh, setup.fp_settings);
	fp_solver.AddModesToDataCollection(dataCollection);
	Array2D<ParGridFunction *> &extra_stress_gf = fp_solver.GetCurrentExtraStress();
	DivergenceMatrixGridFunctionCoefficient *div_extra_stress = new DivergenceMatrixGridFunctionCoefficient(extra_stress_gf);
	for (int i = 0; i < pmesh.Dimension(); ++i)
	{
		for (int j = 0; j < pmesh.Dimension(); ++j)
		{
			dataCollection.RegisterField("tau" + std::to_string(i) + std::to_string(j), extra_stress_gf[i][j]);
		}
	}
	ParGridFunction div_extra_stress_gf(&vfes);
	div_extra_stress_gf.ProjectCoefficient(*div_extra_stress);
	dataCollection.RegisterField("div_tau", &div_extra_stress_gf);

	NavierSolver2 flowsolver(&pmesh, setup);
	flowsolver.AddAccelTerm(div_extra_stress, domain_attr);

	ParGridFunction &velocity_gf = *flowsolver.GetCurrentVelocity();
	ParGridFunction &pressure_gf = *flowsolver.GetCurrentPressure();
	dataCollection.RegisterField("u", &velocity_gf);
	dataCollection.RegisterField("p", &pressure_gf);

	// Coupling to Fokker-Planck eq
	VectorGridFunctionCoefficient u_coeff(&velocity_gf);
	GradientVectorGridFunctionCoefficient grad_u(&velocity_gf);
	MatrixArrayCoefficient dudx_coeff(pmesh.Dimension());
	std::vector<ParGridFunction> dudx_gf(grad_u.GetHeight() * grad_u.GetWidth());
	for (int i = 0; i < grad_u.GetHeight(); ++i)
	{
		for (int j = 0; j < grad_u.GetWidth(); ++j)
		{
			const int idx = i * grad_u.GetWidth() + j;
			dudx_coeff.Set(i, j, new MatrixEntryCoefficient(&grad_u, i, j));
			dudx_gf[idx].SetSpace(&fes);
			dudx_gf[idx].ProjectCoefficient(*dudx_coeff.GetCoeff(i, j));
			dataCollection.RegisterField("du" + std::to_string(i) + "dx" + std::to_string(j), &dudx_gf[idx]);
		}
	}
	fp_solver.AddSpaceAdvectionTerm(domain_attr, u_coeff);
	fp_solver.AddConfigurationAdvectionTerm(domain_attr, dudx_coeff);

	fp_solver.Setup(setup.time_settings.getTimeIntegrator());
	flowsolver.Setup(setup.time_settings.dt);

#if false
	const int vector_size = setup.fp_settings.GetNModes();
	ParFiniteElementSpace fespace(&pmesh, &fec);									   // scalar
	ParFiniteElementSpace v2dfespace(&pmesh, &fec, 2, Ordering::byNODES);			   // 2D vector
	ParFiniteElementSpace block_fespace(&pmesh, &fec, vector_size, Ordering::byNODES); // phi vector
	if (Mpi::Root())
	{
		std::cout << "Number of mode coefficients: " << vector_size << std::endl;
	}
	const int n_dof = fespace.GetVSize();
	const int n_true_dof = fespace.GetTrueVSize();

	// Create offset vector for phi vector, i.e., indices where the next phi_{k,*} starts
	Array<int> block_offsets(vector_size + 1);
	block_offsets[0] = 0;
	for (int i = 1; i < vector_size + 1; ++i)
	{
		block_offsets[i] = block_offsets[i - 1] + n_dof;
	}
	// Create true offset vector for phi vector
	Array<int> true_block_offsets(vector_size + 1);
	true_block_offsets[0] = 0;
	for (int i = 1; i < vector_size + 1; ++i)
	{
		true_block_offsets[i] = true_block_offsets[i - 1] + n_true_dof;
	}

	// block vector
	BlockVector phi_block(block_offsets);
	BlockVector phi0_block(block_offsets);
	BlockVector phi_dt_I_alpha_block(block_offsets);

	phi_block = 0.0;
	phi0_block = 0.0;
	phi_dt_I_alpha_block = 0.0;

	// true block vector for parallelization
	BlockVector phi_true_block(true_block_offsets);
	BlockVector phi0_true_block(true_block_offsets);
	BlockVector phi_dt_I_alpha_true_block(block_offsets);

	phi_true_block = 0.0;
	phi0_true_block = 0.0;
	phi_dt_I_alpha_true_block = 0.0;

	// Needed to project the initial condition
	ParGridFunction phi0(&block_fespace, phi0_block.GetData());
	ParGridFunction phi(&block_fespace, phi_block.GetData());

	// Fokker-Planck
	// TODO: replace by other IC
	VectorFunctionCoefficient phi_IC(vector_size, [](const Vector &, Vector &v)
									 { v = 0.0; v(0) = 1.0; });
	phi0.ProjectCoefficient(phi_IC);
	phi.ProjectCoefficient(phi_IC);

	////////////////////////////////////////////////////////////////////////////
	// Create references to the individual phis
	std::vector<ParGridFunction> phis0(vector_size);
	std::vector<ParGridFunction> phis(vector_size);
	std::vector<ParGridFunction> phis_dt_I_alpha(vector_size);

	for (int i = 0; i < vector_size; i++)
	{
		phis0[i].MakeRef(&fespace, phi_block.GetBlock(i), 0);
		phis[i].MakeRef(&fespace, phi_block.GetBlock(i), 0);
		phis_dt_I_alpha[i].MakeRef(&fespace, phi_dt_I_alpha_block.GetBlock(i), 0);
	}

	for (int i = 0; i < vector_size; i++)
	{
		// TODO correct naming
		dataCollection.RegisterField("phi " + std::to_string(i), &phis[i]);
		dataCollection.RegisterField("phi_dt_I_alpha " + std::to_string(i), &phis_dt_I_alpha[i]);
	}

	for (int i = 0; i < vector_size; i++)
	{
		phis[i].GetTrueDofs(phi_true_block.GetBlock(i));
	}

	// Create vector of block vectors for modes
	std::vector<BlockVector> phi_true_modes(N_kernel, BlockVector(true_block_offsets));
	for (int i = 0; i < param.N_kernel; i++)
	{
		phi_true_modes[i] = 0.0;
	}

	////////////////////////////////////////////////////////////////////////////
	// Add Dirichlet boundary conditions to velocity space restricted to
	// selected attributes on the mesh.
	Array<int> attr(pmesh.bdr_attributes.Max());
	attr = 1;
	naviersolver.AddVelDirichletBC(vel, attr);
	naviersolver.AddAccelTerm(accel, domain_attr);

	////////////////////////////////////////////////////////////////////////////
	// Calculating div_x T
	ParGridFunction &phi0c = phis_dt_I_alpha[0];
	ParGridFunction &phi2c = phis_dt_I_alpha[1];
	ParGridFunction &phi2s = phis_dt_I_alpha[2];

	GradientGridFunctionCoefficient grad_phi0_coeff(&phi0c);
	GradientGridFunctionCoefficient grad_phi2c_coeff(&phi2c);
	GradientGridFunctionCoefficient grad_phi2s_coeff(&phi2s);

	DenseMatrix swap(pmesh.Dimension());
	swap = 0.0;
	swap.Elem(0, 1) = 25.132741228718345;
	swap.Elem(1, 0) = 25.132741228718345;
	MatrixConstantCoefficient swap_coeff(swap);

	DenseMatrix elim_c1(pmesh.Dimension());
	elim_c1 = 0.0;
	elim_c1.Elem(1, 1) = 35.54306350526693;
	MatrixConstantCoefficient elim_c1_coeff(elim_c1);

	DenseMatrix elim_c2(pmesh.Dimension());
	elim_c2 = 0.0;
	elim_c2.Elem(0, 0) = 35.54306350526693;
	MatrixConstantCoefficient elim_c2_coeff(elim_c2);

	ScalarVectorProductCoefficient T_1(25.132741228718345, grad_phi0_coeff);
	MatrixVectorProductCoefficient T_2(swap_coeff, grad_phi0_coeff);
	MatrixVectorProductCoefficient T_3(elim_c1_coeff, grad_phi2c_coeff);
	MatrixVectorProductCoefficient T_4(elim_c2_coeff, grad_phi2s_coeff);

	VectorSumCoefficient T_12(T_1, T_2);
	VectorSumCoefficient T_34(T_3, T_4);

	VectorSumCoefficient T(T_12, T_34, param.scale_T, param.scale_T);

	//	Array<int> domain_attr(pmesh.attributes.Max());
	//	domain_attr = 1;
	//	naviersolver.AddAccelTerm(T, domain_attr);

	GridFunctionCoefficient phi0_coeff(&phi0c);
	GridFunctionCoefficient phi2c_coeff(&phi2c);
	GridFunctionCoefficient phi2s_coeff(&phi2s);

	////////////////////////////////////////////////////////////////////////////
	// Set parameters xi and chi

	ConstantCoefficient one_coeff(1.0);
	ProductCoefficient xi_coeff(1.0, one_coeff);
	ProductCoefficient chi_coeff(1.0, one_coeff);

	////////////////////////////////////////////////////////////////////////////
	// Rational Approximation for the TFNSFP system
	// the values are precomputed in python

	std::vector<double> lambdas = get_lambdas(param.alpha, param.N_kernel);
	std::vector<double> weights = get_weights(param.alpha, param.N_kernel);
	std::vector<double> gammas = get_gammas(param.alpha, param.N_kernel, setup.time_settings.dt);
	double w_inf = get_w_infinity(param.alpha);

	std::cout << "gammas" << std::endl;
	for (auto i : gammas)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;

	std::cout << "weights" << std::endl;
	for (auto i : weights)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;

	std::cout << "lambdas" << std::endl;
	for (auto i : lambdas)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;

	std::cout << "w_inf: " << w_inf << std::endl;

	////////////////////////////////////////////////////////////////////////////
	// Prepare for simulation

	Vector tmp_vector(n_true_dof);
	BlockVector tmp_block_vector(true_block_offsets);
	BlockVector tmp2_block_vector(true_block_offsets);

	// Get velocity and pressure from Navier-Stokes solver
	ParGridFunction &u_gf_NS = *naviersolver.GetCurrentVelocity();
	ParGridFunction &p_gf_NS = *naviersolver.GetCurrentPressure();
	VectorGridFunctionCoefficient u_coeff(&u_gf_NS);

	// ****************************************************************
	// Derivatives of u
	ParFiniteElementSpace u_fes(&pmesh, u_gf_NS.ParFESpace()->FEColl(), 1);

	ParGridFunction dxu1_gf(&u_fes);
	ParGridFunction dyu1_gf(&u_fes);
	ParGridFunction dxu2_gf(&u_fes);
	ParGridFunction dyu2_gf(&u_fes);

	u_gf_NS.GetDerivative(1, 0, dxu1_gf);
	u_gf_NS.GetDerivative(1, 1, dyu1_gf);
	u_gf_NS.GetDerivative(2, 0, dxu2_gf);
	u_gf_NS.GetDerivative(2, 1, dyu2_gf);

	GridFunctionCoefficient d1u1_coeff(&dxu1_gf);
	GridFunctionCoefficient d2u1_coeff(&dyu1_gf);
	GridFunctionCoefficient d1u2_coeff(&dxu2_gf);
	GridFunctionCoefficient d2u2_coeff(&dyu2_gf);

	// For the transformation to the coefficient space
	ParBilinearForm m(&fespace);
	m.AddDomainIntegrator(new MassIntegrator);
	m.Assemble();
	m.Finalize();
	HypreParMatrix *const m_HPM = m.ParallelAssemble();

	BiCGSTABSolver m_solver(MPI_COMM_WORLD);
	m_solver.iterative_mode = false;
	m_solver.SetRelTol(1e-12);
	m_solver.SetMaxIter(1000);
	m_solver.SetPrintLevel(0);
	m_solver.SetOperator(*m_HPM);

	////////////////////////////////////////////////////////////////////////////
	// Output

	ParGridFunction chi_gf(&fespace);
	chi_gf.ProjectCoefficient(chi_coeff);

	ParGridFunction xi_gf(&fespace);
	xi_gf.ProjectCoefficient(xi_coeff);

	ParGridFunction T_gf(&v2dfespace);
	T_gf.ProjectCoefficient(T);

	ParGridFunction C11_gf(&fespace);
	SumCoefficient C11_coeff(phi0_coeff, phi2c_coeff, 25.132741228718345, 35.54306350526693);
	C11_gf.ProjectCoefficient(C11_coeff);

	ParGridFunction C12_gf(&fespace);
	ProductCoefficient C12_coeff(25.132741228718345, phi2s_coeff);
	C12_gf.ProjectCoefficient(C12_coeff);

	ParGridFunction C22_gf(&fespace);
	SumCoefficient C22_coeff(phi0_coeff, phi2c_coeff, 25.132741228718345, 35.54306350526693);
	C22_gf.ProjectCoefficient(C22_coeff);

	phi_true_block = phi0_true_block;
#endif

	////////////////////////////////////////////////////////////////////////////
	// Time loop
	double t = 0.0;
	const double dt = setup.time_settings.dt;
	const double t_final = setup.time_settings.t_final;

	dataCollection.SetCycle(0);
	dataCollection.SetTime(t);
	dataCollection.Save();

	bool last_step = false;
	for (int step = 1; !last_step; ++step)
	{
		if (t + dt >= t_final - dt / 2)
		{
			last_step = true;
		}

		if (Mpi::Root())
		{
			std::cout << "t: " << t << "s / " << t_final << "s - dt: " << dt << std::endl;
		}

		flowsolver.Step(t, dt, step - 1);
		t -= dt; // Due to double time-integration
		fp_solver.Step(t, dt, step);

#if false
		// create operators
		CSS css(fespace, vector_size, true_block_offsets, &u_gf_NS, chi_coeff, xi_coeff, get_beta(param.alpha, param.N_kernel, dt));
		PSS pss(fespace, u_coeff, get_beta(param.alpha, param.N_kernel, dt), param.epsilon);

		// Accumulate fractional derivative of psi
		phi_dt_I_alpha_true_block = 0;
		if (t < 1e-8)
		{
		}
		else
		{
			phi_dt_I_alpha_true_block.Add(-w_inf, phi_true_block);
			for (int k = 0; k < param.N_kernel; ++k)
			{
				phi_dt_I_alpha_true_block.Add(-1, phi_true_modes[k]);
			}
		}

		tmp_block_vector = 0;
		tmp2_block_vector = 0;

		// RHS - inside the operator
		for (int k = 0; k < param.N_kernel; ++k)
		{
			tmp_block_vector.Add(gammas[k], phi_true_modes[k]);
			tmp_block_vector.Add(-1, phi_true_modes[k]);
		}

		// psi_inf(0) = 0
		if (t < 1e-8)
		{
		}
		else
		{
			tmp_block_vector.Add(-w_inf, phi_true_block);
		}

		css.apply_FR(tmp_block_vector, tmp2_block_vector);
		for (int i = 0; i < vector_size; i++)
		{
			m_solver.Mult(tmp2_block_vector.GetBlock(i), tmp_block_vector.GetBlock(i));
		}

		// RHS - outside the operator
		tmp_block_vector.Add(1.0, phi_true_block);

		// SOLVE
		css.solve_Id_minus_theta_FR(tmp_block_vector, phi_true_block);

		tmp_block_vector = 0;
		tmp2_block_vector = 0;

		// RHS - inside the operator
		for (int k = 0; k < param.N_kernel; k++)
		{
			tmp_block_vector.Add(gammas[k], phi_true_modes[k]);
			tmp_block_vector.Add(-1, phi_true_modes[k]);
		}

		// psi_inf(0) = 0
		if (t < 1e-8)
		{
		}
		else
		{
			tmp_block_vector.Add(-w_inf, phi_true_block);
		}

		for (int i = 0; i < vector_size; ++i)
		{
			pss.apply_Fx(tmp_block_vector.GetBlock(i), tmp_vector);
			m_solver.Mult(tmp_vector, tmp_block_vector.GetBlock(i));
		}

		// RHS - outside the operator
		tmp_block_vector.Add(1.0, phi_true_block);

		// SOLVE
		for (int i = 0; i < vector_size; i++)
		{
			pss.solve_Id_minus_beta_Fx(tmp_block_vector.GetBlock(i), phi_true_block.GetBlock(i));
		}

		// mode update
		for (int k = 0; k < param.N_kernel; ++k)
		{
			phi_true_modes[k].Add(weights[k] * dt, phi_true_block);
			phi_true_modes[k] *= gammas[k];
		}

		// Accumulate fractional derivative of psi
		phi_dt_I_alpha_true_block.Add(w_inf, phi_true_block);
		for (int k = 0; k < param.N_kernel; ++k)
		{
			phi_dt_I_alpha_true_block += phi_true_modes[k];
		}
		phi_dt_I_alpha_true_block *= 1 / dt;

		// ****************************************************************
		// Load output

		for (int i = 0; i < vector_size; i++)
		{
			phis[i].Distribute(phi_true_block.GetBlock(i));
			phis_dt_I_alpha[i].Distribute(phi_dt_I_alpha_true_block.GetBlock(i));
		}

		// Project coefficients on grid functions for the output
		T_gf.ProjectCoefficient(T);
		chi_gf.ProjectCoefficient(chi_coeff);
		xi_gf.ProjectCoefficient(xi_coeff);
		C11_gf.ProjectCoefficient(C11_coeff);
		C12_gf.ProjectCoefficient(C12_coeff);
		C22_gf.ProjectCoefficient(C22_coeff);

		div_u_gf.Distribute(div_u_gf.GetTrueDofs());
		T_gf.Distribute(T_gf.GetTrueDofs());
#endif

		if (last_step || (step % output_steps) == 0)
		{
			double err = 0.0;
			if (setup.has_analytic_solution())
			{
				auto *exact = setup.GetAnalyticModes(t);
				auto &sol = fp_solver.GetCurrentModes();
				for (int i = 0; i < sol.size(); ++i)
				{
					double l2 = sol[i].ComputeL2Error(*exact->GetCoeff(i));
					err += l2 * l2;
				}
				err = std::sqrt(err) / sol.size();
			}

			if (Mpi::Root())
			{
				out << "step " << std::setw(6) << step
					<< ",  t = " << std::fixed << std::setprecision(4) << t;
				if (setup.has_analytic_solution())
				{
					out << ",  L2_err = " << std::scientific << std::setprecision(3) << err;
				}
				out << std::defaultfloat << std::endl;
			}

			// update ParGridFunctions for output
			for (int i = 0; i < grad_u.GetHeight(); ++i)
			{
				for (int j = 0; j < grad_u.GetWidth(); ++j)
				{
					const int idx = i * grad_u.GetWidth() + j;
					dudx_gf[idx].ProjectCoefficient(*dudx_coeff.GetCoeff(i, j));
				}
			}
			div_extra_stress_gf.ProjectCoefficient(*div_extra_stress);

			dataCollection.SetCycle(step);
			dataCollection.SetTime(t);
			dataCollection.Save();
		}
	}

	sw_all.Stop();
	double my_rt = sw_all.RealTime();
	double max_rt;
	MPI_Reduce(&my_rt, &max_rt, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh.GetComm());

	fp_solver.PrintTimingData();
	flowsolver.PrintTimingData();

	if (Mpi::Root())
	{
		out << std::endl
			<< std::defaultfloat
			<< "The simulation took " << max_rt << " seconds." << std::endl
			<< std::endl
			<< "Results were written to: " << dataCollection.GetPrefixPath() << pathname << std::endl;
	}

	return EXIT_SUCCESS;
}