#define MFEM_USE_MPI

#include "TensorFokkerPlanckOperator.hpp"

#include "FPOperator.hpp"

using namespace mfem;
using namespace fokker;

Array<int> make_offsets(int fTVS, int dim) {
  Array<int> range;
  if (dim == 2) {
    range = Array<int>({0, fTVS, 2*fTVS, 3*fTVS}); 
  } else {
    range = Array<int>({0, fTVS, 2*fTVS, 3*fTVS, 4*fTVS, 5*fTVS, 6*fTVS}); 
  }
  return range;
}

TensorFokkerPlanckOperator::TensorFokkerPlanckOperator(
    ParFiniteElementSpace &fes, const Array<int> ess_tdofs,
    VectorCoefficient &space_adv_coeff, MatrixArrayCoefficient &conf_adv_coeff,
    const FokkerPlanckSettings &settings)
    : FPOperator(settings.GetNModes(fes.GetMesh()->Dimension()) *
                 fes.GetTrueVSize()),
      dim(fes.GetMesh()->Dimension()),
      settings(settings),
      fespace(fes),
      ess_tdof_list(ess_tdofs),
      Mform(&fes),
      Mform2(&fes),
      Kform(&fes),
      Cform(settings.GetNModes(dim) * (settings.GetNModes(dim) - 1)),
      Cmat(settings.GetNModes(dim) * (settings.GetNModes(dim) - 1)),
      Cmat_old(settings.GetNModes(dim) * (settings.GetNModes(dim) - 1)),
      Dmat(settings.GetNModes(dim) * (settings.GetNModes(dim) - 1)),
      Tmat(settings.GetNModes(dim)),
      T_solver(settings.GetNModes(dim)),
      M_solver(fes.GetComm()),
      T_prec_solver(fes.GetComm()),
      fp_op_solver(fes.GetComm()),
      tmp(fes.GetTrueVSize()),
      block_offsets(make_offsets(fes.GetTrueVSize(), dim)),
      fp_op(block_offsets),
      fp_op_prec(block_offsets),
      btmp(block_offsets) {

  // Mass matrix
  Mform.AddDomainIntegrator(new MassIntegrator);
  Mform.Assemble(0);  // keep sparsity pattern
  Mform.FormSystemMatrix(ess_tdof_list, Mmat);
  
  // Diffusion matrix
  ConstantCoefficient diff_coeff(-settings.space_diff_coeff);  // Diffusion has opposite sign than usual
  Kform.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
  Kform.Assemble(0);                            // keep sparsity pattern
  Kform.FormSystemMatrix(ess_tdof_list, Kmat);  // including BC
  Array<int> empty;
  Kform.FormSystemMatrix(empty, Kmat0);  // without BC

  Cform = nullptr;
  for (auto &C : Cmat) {
    C = new HypreParMatrix;
  }
  for (auto &C : Cmat_old) {
    C = new HypreParMatrix;
  }
  for (auto &D : Dmat) {
    D = new HypreParMatrix;
  }

  // Solvers for dual primal transformationn 
  M_solver.iterative_mode = false;
  M_solver.SetRelTol(settings.rel_tol);
  M_solver.SetAbsTol(settings.abs_tol);
  M_solver.SetMaxIter(settings.max_iter);
  M_solver.SetPrintLevel(settings.print_level_M);
  M_prec.SetType(HypreSmoother::Jacobi);
  M_solver.SetPreconditioner(M_prec);
  M_solver.SetOperator(Mmat);

  // Preconditioner for implicit solver 
  T_prec_solver.iterative_mode = false;
  T_prec_solver.SetRelTol(settings.rel_tol);
  T_prec_solver.SetAbsTol(settings.abs_tol);
  T_prec_solver.SetMaxIter(settings.max_iter);
  T_prec_solver.SetPrintLevel(settings.print_level_M);
  T_prec_prec.SetType(HypreSmoother::Jacobi);
  T_prec_solver.SetPreconditioner(T_prec_prec);
  T_prec_solver.SetOperator(Mmat);

  fp_op_solver.iterative_mode = false;
  fp_op_solver.SetRelTol(settings.rel_tol);
  fp_op_solver.SetAbsTol(settings.abs_tol);
  fp_op_solver.SetMaxIter(settings.max_iter);
  fp_op_solver.SetPrintLevel(settings.print_level_fp);
  fp_op_solver.SetPreconditioner(fp_op_prec);

  for (int k = 0; k < settings.GetNModes(dim); k++) {
    T_solver[k] = new CGSolver(fes.GetComm());
    T_solver[k]->iterative_mode = false;
    T_solver[k]->SetRelTol(1e-6);
    T_solver[k]->SetAbsTol(1e-100);
    T_solver[k]->SetMaxIter(100);
    T_solver[k]->SetPrintLevel(settings.print_level_T);
    T_solver[k]->SetPreconditioner(T_prec_solver);
  }

  Tmat = nullptr;

  // initialize parameter function
  SetParameters(space_adv_coeff, conf_adv_coeff);
}

void TensorFokkerPlanckOperator::MassMult(const Vector &phi, Vector &dphi_dt) const {
  Mmat.Mult(phi, dphi_dt); 
} 

void TensorFokkerPlanckOperator::Mult(const Vector &phi, Vector &dphi_dt) const {
  const int vsize = fespace.TrueVSize();

  // Load phi00 data
  const Vector phi0(phi.GetData(), vsize);

  // Solve 0th mode
  {
    // solution vector
    Vector dphi0_dt(dphi_dt.GetData(), vsize);

    // Apply advection diffusion
    Amat.Mult(phi0, tmp);
    Kmat.AddMult(phi0, tmp);

    // Transform from dual to primal
    M_solver.Mult(tmp, dphi0_dt);

    // Boundary conditions
    dphi0_dt.SetSubVector(ess_tdof_list, 0.0);

    MFEM_VERIFY(M_solver.GetConverged(), "Mass solver did not converge.");
    if (settings.verbose) {
      out << std::setw(8) << "" << std::setw(5) << "Iter " << std::setw(8)
          << "Residual" << std::endl;
      out << std::setw(8) << "phi00  " << std::setw(4)
          << M_solver.GetNumIterations() << " " << std::setw(8)
          << std::setprecision(2) << std::scientific << M_solver.GetFinalNorm()
          << std::defaultfloat << std::endl;
    }
  }

  // Solve other modes
  {
    const BlockVector phis(phi.GetData() + vsize, block_offsets);
    BlockVector dphis_dt(dphi_dt.GetData() + vsize, block_offsets);



    ApplyForwardMode(phi0, phis, btmp);

    for (int i = 0; i < 1; ++i) {
      // Transform blockwise from dual to primal
      M_solver.Mult(btmp.GetBlock(i), dphis_dt.GetBlock(i));
      dphis_dt.GetBlock(i).SetSubVector(ess_tdof_list,
                                        0.0);  // correct? remove essential BCs

      MFEM_VERIFY(M_solver.GetConverged(), "Mass solver did not converge.");
      if (settings.verbose) {
        std::string name = "phi^";
        name += std::to_string(i) + " ";
        out << std::setw(8) << name << std::setw(4)
            << M_solver.GetNumIterations() << " " << std::setw(8)
            << std::setprecision(2) << std::scientific
            << M_solver.GetFinalNorm() << std::defaultfloat << std::endl;
      }
    }
  }
}

void TensorFokkerPlanckOperator::GeneralizedImplicitSolve(const double dtn1, 
                                                          const Vector &phin, 
                                                          const Vector &inFn1, 
                                                          Vector  &phin1) 
{
  // Solve (Id - dtn1 Fn1 - dtn Fn) x = phin + Fn(inFn) + Fn1(inFn1) 

  //****************************************************************  
  // 0th mode 
  const int vsize = fespace.TrueVSize();
  const Vector phi0_n(phin.GetData(), vsize);
  const Vector phi0_inFn1(inFn1.GetData(), vsize);
  Vector phi0_n1(phin1.GetData(), vsize);
 
  // valid only for Neumann boundary conditions 
  phi0_n1 = 1.; 

  //****************************************************************  
  // other modes

  const BlockVector phis_n(phin.GetData() + vsize, block_offsets);
  const BlockVector phis_inFn1(inFn1.GetData() + vsize, block_offsets);
  BlockVector phis_n1(phin1.GetData() + vsize, block_offsets);
  BlockVector zero_bv(block_offsets); zero_bv=0.; 
  BlockVector tmp_bv(block_offsets); 

  // phin term - transform to dual 
  for (int k = 1; k < settings.GetNModes(dim); ++k) {
    Mmat.Mult(phis_n.GetBlock(k-1), btmp.GetBlock(k-1));
  }  

  // Compute contribution of phi0_n1 to phis_n1
  ApplyForwardMode(phi0_n1, zero_bv, tmp_bv);
  btmp.Add(dtn1, tmp_bv); 
  
  // Fn+1 term 
  ApplyForwardMode(phi0_inFn1, phis_inFn1, tmp_bv);
  btmp.Add(1., tmp_bv); 

  // create diagonal of the solver 
  if (!Tmat[1]) {    
    for (int k = 1; k < settings.GetNModes(dim); ++k) {
      delete Tmat[k];
      // add advection diffusion
      Tmat[k] = Add(1.0, Mmat, - dtn1, Kmat);
      Tmat[k]-> Add(- dtn1, Amat);
    } 
    // add configuration operator
    for (int k = 1; k < settings.GetNModes(dim); ++k) {
      Tmat[k]->Add(- dtn1,  *Cmat[1 + (settings.GetNModes(dim) + 1) * (k - 1)]);
    }
  }

  // set diagonal preconditioner
  for (int k = 1; k < settings.GetNModes(dim); ++k) {
    T_solver[k]->SetOperator(*Tmat[k]);
    fp_op_prec.SetDiagonalBlock(k - 1, T_solver[k]);
    fp_op.SetDiagonalBlock(k - 1, Tmat[k]);
  }

  std::vector<int> zero_matrices = {};
  switch (dim) {
    case (2):
      zero_matrices = {7, 10};
      break;
    case (3):
      zero_matrices = {6, 12, 18, 24, 26, 27, 30, 32, 34, 36, 39, 40};
      break;
  }

  // create offdiagonal parts of the solver 
  for (int k = 0; k < settings.GetNModes(dim) - 1; ++k) {
    for (int l = 0; l < settings.GetNModes(dim) - 1; ++l) {
      if (k != l) {
        int Cmat_idx = k * settings.GetNModes(dim) + l + 1;
        // check whether the entry is zero
        if (std::find(std::begin(zero_matrices), std::end(zero_matrices), Cmat_idx) == std::end(zero_matrices)) {
          fp_op.SetBlock(k, l, Cmat[Cmat_idx], -dtn1);
        }
      }
    }
  }

  fp_op_solver.SetOperator(fp_op);
  fp_op_solver.Mult(btmp, phis_n1);

  MFEM_VERIFY(fp_op_solver.GetConverged(),"Implicit solver for phis did not converge.");
  if (settings.verbose) {
    std::string name = "phis";
    out << std::setw(8) << name << std::setw(4)
        << fp_op_solver.GetNumIterations() << " " << std::setw(8)
        << std::setprecision(2) << std::scientific
        << fp_op_solver.GetFinalNorm() << std::defaultfloat << std::endl;
  }
}  

void TensorFokkerPlanckOperator::ImplicitSolve(const double dt,
                                               const Vector &phi,
                                               Vector &dphi_dt) {
  const int vsize = fespace.TrueVSize();
  const Vector phi0(phi.GetData(), vsize);

  if (!Tmat[0] || dt != current_dt) {
    for (int k = 0; k < settings.GetNModes(dim); ++k) {
      delete Tmat[k];
      // add spatial operator
      Tmat[k] = Add(1.0, Mmat, -dt, Amat);
      Tmat[k]->Add(-dt, Kmat);
    }
    // Add configuration operator
    for (int k = 1; k < settings.GetNModes(dim); ++k) {
      Tmat[k]->Add(-dt, *Cmat[1 + (settings.GetNModes(dim) + 1) * (k - 1)]);
    }
    current_dt = dt;
  }

  // Solve 0th mode
  Vector dphi0_dt(dphi_dt.GetData(), vsize);
  // valid only for Neumann boundary conditions 
  dphi0_dt = 0.; 

  dphi0_dt.SetSubVector(ess_tdof_list, 0.0); 

  // Solve other modes
  const BlockVector phis(phi.GetData() + vsize, block_offsets);
  BlockVector dphis_dt(dphi_dt.GetData() + vsize, block_offsets);
  ApplyForwardMode(phi0, phis, btmp);

  // diagonal preconditioner
  for (int k = 1; k < settings.GetNModes(dim); ++k) {
    T_solver[k]->SetOperator(*Tmat[k]);
    // shifted since Tmat[0] is for phi0
    fp_op_prec.SetDiagonalBlock(k - 1, T_solver[k]);
    fp_op.SetDiagonalBlock(k - 1, Tmat[k]);
  }

  std::vector<int> zero_matrices = {};
  switch (dim) {
    case (2):
      zero_matrices = {7, 10};
      break;
    case (3):
      zero_matrices = {6, 12, 18, 24, 26, 27, 30, 32, 34, 36, 39, 40};
      break;
  }

  for (int k = 0; k < settings.GetNModes(dim) - 1; ++k) {
    for (int l = 0; l < settings.GetNModes(dim) - 1; ++l) {
      if (k != l) {
        // check whether the entry is zero
        int Cmat_idx = k * settings.GetNModes(dim) + l + 1;
        if (std::find(std::begin(zero_matrices), std::end(zero_matrices),
                      Cmat_idx) == std::end(zero_matrices)) {
          fp_op.SetBlock(k, l, Cmat[Cmat_idx], -dt);
        }
      }
    }
  }

  fp_op_solver.SetOperator(fp_op);
  fp_op_solver.Mult(btmp, dphis_dt);

  for (int k = 0; k < settings.GetNModes(dim) - 1; ++k) {
    dphis_dt.GetBlock(k).SetSubVector(ess_tdof_list, 0.0);
  }

  MFEM_VERIFY(fp_op_solver.GetConverged(),
              "Implicit solver for phis did not converge.");

  if (settings.verbose) {
    std::string name = "phis";
    out << std::setw(8) << name << std::setw(4)
        << fp_op_solver.GetNumIterations() << " " << std::setw(8)
        << std::setprecision(2) << std::scientific
        << fp_op_solver.GetFinalNorm() << std::defaultfloat << std::endl;
  }
}


void TensorFokkerPlanckOperator::SetParameters(
    VectorCoefficient &space_adv_coeff,
    MatrixArrayCoefficient &conf_adv_coeff) {
    
  // Update spatial advection form
  delete Aform;
  Array<int> empty; // no BC 
  Aform = new ParBilinearForm(&fespace);
  Aform->AddDomainIntegrator(new ConvectionIntegrator(space_adv_coeff, -1.0));
  Aform->Assemble(0);  // keep sparsity pattern of all matrices the same
  Aform->FormSystemMatrix(ess_tdof_list, Amat);
  Aform->FormSystemMatrix(empty, Amat0); 

  // Note that in this formulation xi = chi = 1/De
  double chi = settings.conf_diff_coeff;
  double a = settings.hermite_scaling;

  // Update coupling forms
  std::vector<SumCoefficient> Ccoeff;
  ConstantCoefficient zero_coeff(0.);

  SumCoefficient tr_conv_adv_coeff_12(*conf_adv_coeff.GetCoeff(0, 0),
                                      *conf_adv_coeff.GetCoeff(1, 1));
  SumCoefficient tr_conv_adv_coeff_13(*conf_adv_coeff.GetCoeff(0, 0),
                                      *conf_adv_coeff.GetCoeff(2, 2));
  SumCoefficient tr_conv_adv_coeff_23(*conf_adv_coeff.GetCoeff(1, 1),
                                      *conf_adv_coeff.GetCoeff(2, 2));

  switch (dim) {
    case (2): {
      // C00
      Ccoeff.emplace_back(*conf_adv_coeff.GetCoeff(0, 1),
                          *conf_adv_coeff.GetCoeff(1, 0));
      // C01
      Ccoeff.emplace_back(-2 * chi, tr_conv_adv_coeff_12);
      // C02
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 1));
      // C03
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 0));

      // C10
      Ccoeff.emplace_back(2 * a * a * chi - chi, *conf_adv_coeff.GetCoeff(1, 1),
                          sqrt(2), sqrt(2));
      // C11
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 0), 0, sqrt(2));
      // C12
      Ccoeff.emplace_back(-chi, *conf_adv_coeff.GetCoeff(1, 1), 2, 2);
      // C13 // TODO: DON'T CONSIDER AT ALL, REMOVE
      Ccoeff.emplace_back(0, zero_coeff);

      // C20
      Ccoeff.emplace_back(2 * a * a * chi - chi, *conf_adv_coeff.GetCoeff(0, 0),
                          sqrt(2), sqrt(2));
      // C21
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 1), 0, sqrt(2));
      // C22 // TODO: DON'T CONSIDER AT ALL, REMOVE
      Ccoeff.emplace_back(0, zero_coeff);
      // C23
      Ccoeff.emplace_back(-chi, *conf_adv_coeff.GetCoeff(0, 0), 2, 2);
      break;
    }

    case (3): {
      // phi000
      // phi011, phi101, phi110
      // phi002, phi020, phi200

      // C00 C01 C02 C03 C04 C05 C06 for phi011
      // C10 C11 C12 C13 C14 C15 C16 for phi101
      // C20 C21 C22 C23 C24 C25 C26 ...
      // C30 C31 C32 C33 C34 C35 C36
      // C40 C41 C42 C43 C44 C45 C46
      // C50 C51 C52 C53 C54 C55 C56

      // for phi011
      // C00
      Ccoeff.emplace_back(*conf_adv_coeff.GetCoeff(1, 2),
                          *conf_adv_coeff.GetCoeff(2, 1));
      // C01
      Ccoeff.emplace_back(-2 * chi, tr_conv_adv_coeff_23);
      // C02
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 0));
      // C03
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(2, 0));
      // C04
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 2), 0, sqrt(2));
      // C05
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(2, 1), 0, sqrt(2));
      // C06
      Ccoeff.emplace_back(0, zero_coeff);

      // for phi101
      // C10
      Ccoeff.emplace_back(*conf_adv_coeff.GetCoeff(0, 2),
                          *conf_adv_coeff.GetCoeff(2, 0));
      // C11
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 1));
      // C12
      Ccoeff.emplace_back(-2 * chi, tr_conv_adv_coeff_13);
      // C13
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(2, 1));
      // C14
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 2), 0, sqrt(2));
      // C15
      Ccoeff.emplace_back(0, zero_coeff);
      // C16
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(2, 0), 0, sqrt(2));

      // for phi110
      // C20
      Ccoeff.emplace_back(*conf_adv_coeff.GetCoeff(0, 1),
                          *conf_adv_coeff.GetCoeff(1, 0));
      // C21
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 2));
      // C22
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 2));
      // C23
      Ccoeff.emplace_back(-2 * chi, tr_conv_adv_coeff_12);
      // C24
      Ccoeff.emplace_back(0, zero_coeff);
      // C25
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 1), 0, sqrt(2));
      // C26
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 0), 0, sqrt(2));

      // for phi002
      // C30
      Ccoeff.emplace_back(2 * a * a * chi - chi, *conf_adv_coeff.GetCoeff(2, 2),
                          sqrt(2), sqrt(2));
      // C31
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(2, 1), 0, sqrt(2));
      // C32
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(2, 0), 0, sqrt(2));
      // C33
      Ccoeff.emplace_back(0, zero_coeff);
      // C34
      Ccoeff.emplace_back(-chi, *conf_adv_coeff.GetCoeff(2, 2), 2, 2);
      // C35
      Ccoeff.emplace_back(0, zero_coeff);
      // C36
      Ccoeff.emplace_back(0, zero_coeff);

      // for phi020
      // C40
      Ccoeff.emplace_back(2 * a * a * chi - chi, *conf_adv_coeff.GetCoeff(1, 1),
                          sqrt(2), sqrt(2));
      // C41
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 2), 0, sqrt(2));
      // C42
      Ccoeff.emplace_back(0, zero_coeff);
      // C43
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(1, 0), 0, sqrt(2));
      // C44
      Ccoeff.emplace_back(0, zero_coeff);
      // C45
      Ccoeff.emplace_back(-chi, *conf_adv_coeff.GetCoeff(1, 1), 2, 2);
      // C46
      Ccoeff.emplace_back(0, zero_coeff);

      // for phi200
      // C50
      Ccoeff.emplace_back(2 * a * a * chi - chi, *conf_adv_coeff.GetCoeff(0, 0),
                          sqrt(2), sqrt(2));
      // C51
      Ccoeff.emplace_back(0, zero_coeff);
      // C52
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 2), 0, sqrt(2));
      // C53
      Ccoeff.emplace_back(0, *conf_adv_coeff.GetCoeff(0, 1), 0, sqrt(2));
      // C54
      Ccoeff.emplace_back(0, zero_coeff);
      // C55
      Ccoeff.emplace_back(0, zero_coeff);
      // C56
      Ccoeff.emplace_back(-chi, *conf_adv_coeff.GetCoeff(0, 0), 2, 2);
      break;
    }

    default:
      MFEM_ABORT("Only 2 and 3 dimensions are allowed.")
      break;
  }

  for (int i = 0; i < settings.GetNModes(dim) * (settings.GetNModes(dim) - 1);
       i++) {
    delete Cform[i];
    Cform[i] = new ParBilinearForm(&fespace);
    Cform[i]->AddDomainIntegrator(new MassIntegrator(Ccoeff[i]));
    Cform[i]->Assemble(0);  // keep sparsity pattern
    Cform[i]->FormSystemMatrix(ess_tdof_list, *Cmat[i]);
  } 

  // Delete implicit time-stepping operators for rebuild
  for (auto *T : Tmat) {
    delete T;
  }
  Tmat = nullptr;

}

// this is applying the diagonal of the operator
void TensorFokkerPlanckOperator::ApplyAdvectionDiffusion(const Vector &x,
                                                         Vector &y) const {
  Amat.Mult(x, y);
  Kmat.AddMult(x, y);
}

// this is applying RHS
void TensorFokkerPlanckOperator::ApplyForwardMode(const Vector &phi0,
                                                  const BlockVector &phi,
                                                  BlockVector &y) const {
  switch (dim) {
    case (2): {
      auto &phi11 = phi.GetBlock(0);
      auto &phi02 = phi.GetBlock(1);
      auto &phi20 = phi.GetBlock(2);

      auto &y11 = y.GetBlock(0);
      auto &y02 = y.GetBlock(1);
      auto &y20 = y.GetBlock(2);

      // phi 11
      ApplyAdvectionDiffusion(phi11, y11);
      Cmat[0]->AddMult(phi0, y11);
      Cmat[1]->AddMult(phi11, y11);
      Cmat[2]->AddMult(phi02, y11);
      Cmat[3]->AddMult(phi20, y11);

      // phi 02
      ApplyAdvectionDiffusion(phi02, y02);
      Cmat[4]->AddMult(phi0, y02);
      Cmat[5]->AddMult(phi11, y02);
      Cmat[6]->AddMult(phi02, y02);
      // Cmat[7]->AddMult(phi20, y02); // zero matrix

      // phi 20
      ApplyAdvectionDiffusion(phi20, y20);
      Cmat[8]->AddMult(phi0, y20);
      Cmat[9]->AddMult(phi11, y20);
      // Cmat[10]->AddMult(phi02, y20); // zero matrix
      Cmat[11]->AddMult(phi20, y20);
      break;
    }
    case (3): {
      auto &phi011 = phi.GetBlock(0);
      auto &phi101 = phi.GetBlock(1);
      auto &phi110 = phi.GetBlock(2);
      auto &phi002 = phi.GetBlock(3);
      auto &phi020 = phi.GetBlock(4);
      auto &phi200 = phi.GetBlock(5);

      auto &y011 = y.GetBlock(0);
      auto &y101 = y.GetBlock(1);
      auto &y110 = y.GetBlock(2);
      auto &y002 = y.GetBlock(3);
      auto &y020 = y.GetBlock(4);
      auto &y200 = y.GetBlock(5);

      // phi 011
      ApplyAdvectionDiffusion(phi011, y011);
      Cmat[0]->AddMult(phi0, y011);
      Cmat[1]->AddMult(phi011, y011);
      Cmat[2]->AddMult(phi101, y011);
      Cmat[3]->AddMult(phi110, y011);
      Cmat[4]->AddMult(phi002, y011);
      Cmat[5]->AddMult(phi020, y011);
      Cmat[6]->AddMult(phi200, y011);

      // phi 101
      ApplyAdvectionDiffusion(phi101, y101);
      Cmat[7]->AddMult(phi0, y101);
      Cmat[8]->AddMult(phi011, y101);
      Cmat[9]->AddMult(phi101, y101);
      Cmat[10]->AddMult(phi110, y101);
      Cmat[11]->AddMult(phi002, y101);
      Cmat[12]->AddMult(phi020, y101);
      Cmat[13]->AddMult(phi200, y101);

      // phi 110
      ApplyAdvectionDiffusion(phi110, y110);
      Cmat[14]->AddMult(phi0, y110);
      Cmat[15]->AddMult(phi011, y110);
      Cmat[16]->AddMult(phi101, y110);
      Cmat[17]->AddMult(phi110, y110);
      Cmat[18]->AddMult(phi002, y110);
      Cmat[19]->AddMult(phi020, y110);
      Cmat[20]->AddMult(phi200, y110);

      // phi 002
      ApplyAdvectionDiffusion(phi002, y002);
      Cmat[21]->AddMult(phi0, y002);
      Cmat[22]->AddMult(phi011, y002);
      Cmat[23]->AddMult(phi101, y002);
      Cmat[24]->AddMult(phi110, y002);
      Cmat[25]->AddMult(phi002, y002);
      Cmat[26]->AddMult(phi020, y002);
      Cmat[27]->AddMult(phi200, y002);
      // phi 020
      ApplyAdvectionDiffusion(phi020, y020);
      Cmat[28]->AddMult(phi0, y020);
      Cmat[29]->AddMult(phi011, y020);
      Cmat[30]->AddMult(phi101, y020);
      Cmat[31]->AddMult(phi110, y020);
      Cmat[32]->AddMult(phi002, y020);
      Cmat[33]->AddMult(phi020, y020);
      Cmat[34]->AddMult(phi200, y020);

      // phi 200
      ApplyAdvectionDiffusion(phi200, y200);
      Cmat[35]->AddMult(phi0, y200);
      Cmat[36]->AddMult(phi011, y200);
      Cmat[37]->AddMult(phi101, y200);
      Cmat[38]->AddMult(phi110, y200);
      Cmat[39]->AddMult(phi002, y200);
      Cmat[40]->AddMult(phi020, y200);
      Cmat[41]->AddMult(phi200, y200);
      break;
    }
    default:
      MFEM_ABORT("Only 2 and 3 dimensions are allowed.")
      break;
  }
}

TensorFokkerPlanckOperator::~TensorFokkerPlanckOperator() {
  delete Aform;

  for (auto *C : Cform) {
    delete C;
  }

  for (auto *C : Cmat) {
    delete C;
  }

  for (auto *T : Tmat) {
    delete T;
  }

  for (auto *solver : T_solver) {
    delete solver;
  }
}
