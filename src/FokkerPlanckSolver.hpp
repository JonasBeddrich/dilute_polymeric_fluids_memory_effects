#pragma once

#include "mfem.hpp"

#include "TensorFokkerPlanckOperator.hpp"

#define FOKKER_PLANCK_VERSION 0.1

namespace mfem
{
	namespace fokker
	{
	
		/// Container for a Dirichlet boundary condition of the phis.
		class DirichletBC_T
		{
		public:
			DirichletBC_T(Array<int> attr, VectorArrayCoefficient &coeff)
				: attr(attr), coeff(coeff) {}

			Array<int> attr;
			VectorArrayCoefficient &coeff;
		};

		/// Container for the boundary attributes which correspond to an
		/// outflow boundary condition of the phis.
		using OutflowBC_T = Array<int>;

		/// Container for the spatial advection term
		class SpatialAdvectionTerm_T
		{
		public:
			/// Class for a spatial advection term.
			SpatialAdvectionTerm_T(Array<int> attr, VectorCoefficient &coeff)
				: attr(attr), coeff(coeff) {}

			/// Defines the domain part where to apply the advection
			Array<int> attr;
			/// The spatial advection coefficient
			VectorCoefficient &coeff;
		};

		/// Container for a configuration advection term
		class ConfigurationAdvectionTerm_T
		{
		public:
			/// Class for a configuration advection term.
			ConfigurationAdvectionTerm_T(Array<int> attr, MatrixArrayCoefficient &coeff)
				: attr(attr), coeff(coeff) {}

			/// Defines the domain part where to apply the advection
			Array<int> attr;
			/// The configuration advection coefficients
			MatrixArrayCoefficient &coeff;
		};

		class FokkerPlanckSolver
		{
		public:
			
			FokkerPlanckSolver(ParMesh &mesh, FokkerPlanckSettings &settings);

			/// Initialize operators and solvers. Assumes ownership.
			void Setup(ODESolver *solver = nullptr);

			void Step(double &time, double dt, int cur_step, bool provisional = false);

			/// Return a reference to the provisional extra-stress ParGridFunction.
			Array2D<ParGridFunction *> &GetProvisionalExtraStress() { return extra_stress_next_gf; }
			/// Return a reference to the provisional extra-stress ParGridFunction.
			const Array2D<ParGridFunction *> &GetProvisionalExtraStress() const { return extra_stress_next_gf; }

			/// Return a reference to the current extra-stress ParGridFunction.
			Array2D<ParGridFunction *> &GetCurrentExtraStress() { return extra_stress_gf; }
			/// Return a reference to the current extra-stress ParGridFunction.
			const Array2D<ParGridFunction *> &GetCurrentExtraStress() const { return extra_stress_gf; }

			/// Return (a reference) to the provisional vector of modes as ParGridFunction.
			std::vector<ParGridFunction> &GetProvisionalModes() { return modes_next_gf; }
			/// Return (a reference) to the provisional vector of modes as ParGridFunction.
			const std::vector<ParGridFunction> &GetProvisionalModes() const { return modes_next_gf; }

			/// Return (a reference) to the current vector of modes as ParGridFunction.
			std::vector<ParGridFunction> &GetCurrentModes() { return modes_gf; }
			/// Return (a reference) to the current vector of modes as ParGridFunction.
			const std::vector<ParGridFunction> &GetCurrentModes() const { return modes_gf; }

			/// Add a Dirichlet boundary condition to all modes.
			void AddDirichletBC(Array<int> &attr, VectorArrayCoefficient &coeff);

			/// Add an outflow boundary condition to all modes.
			void AddOutflowBC(const Array<int> &attr);

			/// Add a spatial advection term to the equation.
			void AddSpaceAdvectionTerm(Array<int> &attr, VectorCoefficient &coeff);

			/// Add a configuration advection term to the equation.
			void AddConfigurationAdvectionTerm(Array<int> &attr, MatrixArrayCoefficient &coeff);

			/// Add all next modes to the data collection for output
			void AddModesToDataCollection(DataCollection &dc);

			void PrintTimingData();

			/// Rotate entries in the time step and solution history arrays.
			void UpdateTimestepHistory(double dt);

			virtual ~FokkerPlanckSolver();

		protected:
			/// Print information about the Fokker-Planck version.
			void PrintInfo();

			/// Computes the next extra-stress based on the modes.
			void ComputeExtraStress();

			/// Settings for everything.
			FokkerPlanckSettings settings;

			/// The parallel mesh.
			ParMesh &pmesh;

			/// Phi \f$ H^1 \f$ finite element collection.
			const H1_FECollection fec;
			/// Scalar \f$ H^1 \f$ finite element space for individual modes.
			ParFiniteElementSpace fes;

			/// Offsets for (mode) block structure of dofs
			Array<int> offsets;
			/// Block vector of all current modes (all dofs for BCs)
			BlockVector modes;
			/// Block vector of all next modes (all dofs for BCs)
			BlockVector modes_next;
			
			/// Vector of individual GridFunctions for the current modes
			std::vector<ParGridFunction> modes_gf;
			/// Vector of individual GridFunctions for the next modes
			std::vector<ParGridFunction> modes_next_gf;

			/// GridFunctions for the current extra-stress
			Array2D<ParGridFunction *> extra_stress_gf;
			/// GridFunctions for the next extra-stress
			Array2D<ParGridFunction *> extra_stress_next_gf;

			/// The Fokker-Plank operator to apply in each time step
			FPOperator *fp_operator = nullptr;

			/// The ODE solver to apply in each step
			ODESolver *ode_solver = nullptr;

			/// Boundary attributes where Dirichlet BCs are applied
			Array<int> dirichlet_attr;
			/// DOFs to which Dirichlet BCs are applied
			Array<int> dirichlet_tdof;
			// Bookkeeping for Dirichlet BCs.
			std::vector<DirichletBC_T> dirichlet_bcs;

			// Boundary attributes where outflow BCs are applied
			Array<int> outflow_attr;

			/// Bookkeeping for spatial advection terms.
			std::vector<SpatialAdvectionTerm_T> spatial_adv_terms;

			/// Bookkeeping for configuration advection terms.
			std::vector<ConfigurationAdvectionTerm_T> configuration_adv_terms;

			/// current time-step number
			int cur_step = 0;

			// Timers.
			StopWatch sw_setup, sw_step;
		};
	}
}