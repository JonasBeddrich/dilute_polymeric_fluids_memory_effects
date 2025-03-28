#pragma once

#include "FPOperator.hpp"
#include "FokkerPlanckSettings.hpp"
#include "mfem.hpp"

namespace mfem {
namespace fokker {
/**
 * @brief Construct a new Fokker-Planck time dependent operator.
 *
 * Applying a hermite spectral method in configuration space
 * the Fokker-Planck equation in space can be written as:
 *
 *   |M 0 0 0| |dphi_{00}/dt| = |A+K 0       0       0      | |phi_{00}|
 *   |0 M 0 0| |dphi_{11}/dt| = |C00 A+K+C01 C02     C03    | |phi_{11}|
 *   |0 0 M 0| |dphi_{02}/dt| = |C10 C11     A+K+C12 C13    | |phi_{02}|
 *   |0 0 0 M| |dphi_{20}/dt| = |C20 C21     C22     A+K+C23| |phi_{20}|
 *
 * where phi are the spectral mode of the 2D, M is the mass matrix, A is
 * the advection operator, K is the diffusion operator. The matrices C_m
 * are depending on @a conf_diff_coeff and @a space_adv_coeff.
 *
 * @param fes The ParFiniteElementSpace the solution is defined on
 * @param ess_tdofs All essential true dofs (relevant for H1 FESpace)
 * @param space_adv_coeff The spatial advection coefficient
 * @param conf_adv_coeff The configuration advection coefficient matrix
 * @param settings The parameters and options for the operator
 */
class TensorFokkerPlanckOperator : public FPOperator {
 public:
  TensorFokkerPlanckOperator(ParFiniteElementSpace &fes,
                             const Array<int> ess_tdofs,
                             VectorCoefficient &space_adv_coeff,
                             MatrixArrayCoefficient &conf_adv_coeff,
                             const FokkerPlanckSettings &settings);

  /// Forward application of the Fokker-Planck operator,
  /// i.e., dphi_dt = RHS(phi, t), for all modes.
  virtual void Mult(const Vector &phi, Vector &dphi_dt) const override;


  virtual void MassMult(const Vector &phi, Vector &dphi_dt) const;  


  virtual void GeneralizedImplicitSolve(const double dtn1, 
                                        const Vector &inFn, 
                                        const Vector &inFn1, 
                                        Vector  &dphi_dt) override;  

  /// Solve the implicit equation y = RHS(phi + dt*y, t) for the
  /// unknown y = dphi_dt, i.e. for all modes.
  virtual void ImplicitSolve(const double dt, const Vector &phi,
                             Vector &dphi_dt) override;

  /// Update the Fokker-Planck operators.
  /// @param space_adv_coeff spatial advection vector coefficient (velocity)
  /// @param conf_adv_coeff configuration advection coefficient matrix
  virtual void SetParameters(VectorCoefficient &space_adv_coeff,
                             MatrixArrayCoefficient &conf_adv_coeff);

  virtual ~TensorFokkerPlanckOperator();

 protected:
  /// Application of the advection-diffusion-reaction operator
  /// y = (A + K) * x to a single vector without setting BCs.
  void ApplyAdvectionDiffusion(const Vector &x, Vector &y) const;

  /// Application of the forward Fokker-Planck operator the modes
  /// without setting BCs.
  void ApplyForwardMode(const Vector &phi0, const BlockVector &phi,
                        BlockVector &y) const;

  /// FE space for Fokker-Planck operator
  ParFiniteElementSpace &fespace;
  /// Essential true dof array for eliminating Dirichlet BCs.
  const Array<int> ess_tdof_list;

  /// Configuration of the operator
  const FokkerPlanckSettings &settings;

  /// Time-step size for ImplicitSolve
  double current_dt = -1.0;

  /// Spatial dimension
  const int dim;

  /// first iteration 
  bool first_iteration = true; 

  /// Mass form
  ParBilinearForm Mform;
  ParBilinearForm Mform2;
  /// Diffusion form
  ParBilinearForm Kform;
  /// Advection form
  ParBilinearForm *Aform = nullptr;
  /// Coupling forms for entries in conf_adv_coeff
  Array<ParBilinearForm *> Cform;

  /// Mass matrix
  HypreParMatrix Mmat;
  HypreParMatrix Mmat2;
  /// Diffusion matrix
  HypreParMatrix Kmat;
  /// Diffusion matrix without BC
  HypreParMatrix Kmat0;
  /// Advection matrix
  HypreParMatrix Amat;
  HypreParMatrix *Amat_old;
  /// Advection matrix without BC
  HypreParMatrix Amat0;
  HypreParMatrix *Amat0_old;
  /// Coupling matrices for entries in conf_adv_coeff
  Array<HypreParMatrix *> Cmat;
  Array<HypreParMatrix *> Cmat_old;
  /// Combined (n and n+1) coupling matrices for entries in conf_adv_coeff
  Array<HypreParMatrix *> Dmat;
  /// Combined matrices T = M - dt * (A + K + C) for ImplicitSolve
  Array<HypreParMatrix *> Tmat;

  /// Mass matrix Krylov solver
  CGSolver M_solver;
  /// Mass matrix Krylov solver for preconditioning 
  CGSolver T_prec_solver; 
  /// Preconditioner for mass matrix Krylov solver
  HypreSmoother M_prec;
  /// Preconditioner for mass matric solver for preconditioning 
  HypreSmoother T_prec_prec;
  /// Implicit solver for T
  Array<CGSolver *> T_solver;
  /// Array of offsets for the 3x3 block system
  Array<int> block_offsets;
  /// Block operator for the 3x3 Fokker-Planck system
  BlockOperator fp_op;
  /// Implicit solver for 3x3 block system
  CGSolver fp_op_solver;
  /// Preconditioner for the 3x3 block system based on the T
  BlockDiagonalPreconditioner fp_op_prec;

  /// Auxiliary vectors
  mutable Vector tmp;
  mutable BlockVector btmp;
  mutable BlockVector btmp2;
};
}  // namespace fokker
}  // namespace mfem