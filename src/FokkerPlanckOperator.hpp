#pragma once

#include "mfem.hpp"
#include "FokkerPlanckSettings.hpp"
#include "FPOperator.hpp"

namespace mfem
{
	namespace fokker
	{
		/**
		 * @brief Construct a new Fokker-Planck time dependent operator.
		 *
		 * After application of the 2D spherical harmonics in configuration space,
		 * the Fokker-Planck equation in space can be written as:
		 *
		 *   |M 0| |dphi_{c,m}/dt| = |A+K-R_m  C_m  | |phi_{c,m}| + |F_m phi_{c,m-2}|
		 *   |0 M| |dphi_{s,m}/dt| = | -C_m  A+K-R_m| |phi_{s,m}| + |G_m phi_{s,m-2}|
		 *
		 * where phi_m is the vector representing the kth coefficients of the 2D
		 * spherical harmonics, M is the mass matrix, A is the advection operator,
		 * K is the diffusion operator. The reaction operator R_m and the coupling
		 * operator C_m are depending on @a conf_diff_coeff, while F_m and G_m are
		 * lower order coupling operators depending on @a conf_diff_coeff.
		 *
		 * @todo extend explanation and add exact formulas
		 *
		 * @param fes The ParFiniteElementSpace the solution is defined on
		 * @param ess_tdofs All essential true dofs (relevant for H1 FESpace)
		 * @param space_adv_coeff The spatial advection coefficient
		 * @param conf_adv_coeff The configuration advection coefficient matrix
		 * @param settings The parameters and options for the operator
		 */
		class FokkerPlanckOperator : public FPOperator
		{
		public:

			FokkerPlanckOperator(ParFiniteElementSpace &fes,
								 const Array<int> ess_tdofs,
								 VectorCoefficient &space_adv_coeff,
								 MatrixArrayCoefficient &conf_adv_coeff,
								 const FokkerPlanckSettings &settings);

			/// Forward application of the Fokker-Planck operator,
			/// i.e., dphi_dt = RHS(phi, t), for all modes.
			virtual void Mult(const Vector &phi, Vector &dphi_dt) const override;

			/// Solve the implicit equation y = RHS(phi + dt*y, t) for the
			/// unknown y = dphi_dt, i.e. for all modes.
			virtual void ImplicitSolve(const double dt,
									   const Vector &phi,
									   Vector &dphi_dt) override;

			virtual void AddMultNonlinear(const Vector &phi, Vector &dphi_dt) const {} 


			/// Update the Fokker-Planck operators.
			/// @param space_adv_coeff spatial advection vector coefficient (velocity)
			/// @param conf_adv_coeff configuration advection coefficient matrix
			void SetParameters(VectorCoefficient &space_adv_coeff,
							   MatrixArrayCoefficient &conf_adv_coeff);

			virtual ~FokkerPlanckOperator();

		protected:
			/// Application of the m-th mode advection-diffusion-reaction operator
			/// y = (A + K - R_m) * x to a single vector without setting BCs.
			void ApplyAKRm(const Vector &x, const double r_m, Vector &y) const;

			/// Application of the forward Fokker-Planck operator to one mode pair,
			/// i.e, y = RHS(phi_m, phi_m-1, t) for one mode pair y = (y_c, y_s)
			/// without setting BCs.
			void ApplyForwardMode(const BlockVector &phi_m, const BlockVector &phi_mm1, const int m, BlockVector &y) const;

			/// Returns the the reaction constant for the given mode (pair)
			/// TODO: implement more than 2D FENE
			double GetReactionCoeff(const int mode) const;

			/// FE space for Fokker-Planck operator
			ParFiniteElementSpace &fespace;
			/// Essential true dof array for eliminating Dirichlet BCs.
			const Array<int> ess_tdof_list;

			/// Configuration of the operator
			const FokkerPlanckSettings &settings;

			/// Time-step size for ImplicitSolve
			double current_dt = -1.0;

			/// Mass form
			ParBilinearForm Mform;
			/// Diffusion form
			ParBilinearForm Kform;
			/// Advection form
			ParBilinearForm *Aform = nullptr;
			/// Coupling forms for entries in conf_adv_coeff
			Array<ParBilinearForm *> Cform;

			/// Mass matrix
			HypreParMatrix Mmat;
			/// Diffusion matrix
			HypreParMatrix Kmat;
			/// Diffusion matrix without BC
			HypreParMatrix Kmat0;
			/// Advection matrix
			HypreParMatrix Amat;
			/// Advection matrix without BC
			HypreParMatrix Amat0;
			/// Coupling matrices for entries in conf_adv_coeff
			Array<HypreParMatrix *> Cmat;
			/// Combined matrices T_k = M - dt * (A - K - R_k) for ImplicitSolve
			Array<HypreParMatrix *> Tmat;

			/// Mass matrix Krylov solver
			CGSolver M_solver;
			/// Mass matrix preconditioner
			HypreSmoother M_prec;
			/// Implicit solver for T_k
			CGSolver T_solver;
			/// Array of offsets for the 2x2 block system
			Array<int> block_offsets;
			/// Block operator for the 2x2 Fokker-Planck system
			BlockOperator fp_op;
			/// Implicit solver for 2x2 block system
			CGSolver fp_op_solver;
			/// Preconditioner for the 2x2 block system based on the T_k
			BlockDiagonalPreconditioner fp_op_prec;

			/// Auxiliary vectors
			mutable Vector tmp;
			mutable BlockVector btmp;
			mutable BlockVector btmp2;
		};
	}
}