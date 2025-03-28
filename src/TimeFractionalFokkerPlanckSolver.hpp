#define MFEM_USE_MPI

#pragma once

#include "mfem.hpp"

#include "FokkerPlanckOperator.hpp"
#include "TensorFokkerPlanckOperator.hpp"
#include "FokkerPlanckSolver.hpp"

#define FOKKER_PLANCK_VERSION 0.1

namespace mfem
{
	namespace fokker
	{
	
		class TimeFractionalFokkerPlanckSolver : public FokkerPlanckSolver 
		{
		public:
			/**
			 * @brief Initialize data structures, set FE space order and coefficients.
			 *
			 * @param mesh The ParMesh can be a linear or curved parallel mesh.
			 * @param settings All other settings and parameters (dimension-less)
			 */
			TimeFractionalFokkerPlanckSolver(ParMesh &mesh, FokkerPlanckSettings &settings, double alpha);

			/// Initialize operators and solvers. Assumes ownership.	
			void Setup();

			// /**
			//  * @param time The current time
			//  * @param dt The time-step size
			//  * @param cur_step The current time step
			//  * @param provisional Whether to automatically accept the computed time step.
			//  */

			void Step(double &time, double dt, int cur_step, bool provisional = false);

			/// Return a reference to the extrapolated extra-stress ParGridFunction.
			Array2D<ParGridFunction *> &GetExtrapolatedExtraStress() { return extra_stress_np2_gf; }
			/// Return a reference to the extrapolated extra-stress ParGridFunction.
			const Array2D<ParGridFunction *> &GetExtrapolatedExtraStress() const { return extra_stress_np2_gf; }
			
			void SetModeEquationParameters(double dt); 
			
			void SetInitialConditions2D(Coefficient &phi00, Coefficient &phi02, Coefficient &phi11, Coefficient &phi20); 

			void UpdateTimestepHistory(double dt); 

			void ExtrapolateExtraStress(); 

			void SetWeightsPoles(); 
			
			virtual ~TimeFractionalFokkerPlanckSolver();

		protected:

			/// Computes the next extra-stress based on the fractional derivative of the modes.
			void ComputeExtraStress();

			void UpdateTimefractionalDerivative(double dt); 

			double alpha; 
			int m = 20; 
			bool non_zero_initial = false; 

			BlockVector inFn; 
			BlockVector inFn1; 
			BlockVector modes_prev; 
			BlockVector dt_1malpha_modes;

			mutable BlockVector tmp; 
			mutable BlockVector tmp2; 
			mutable BlockVector rhs; 

			/// Block Vector for all the fractional modes of the modes at the current time 
			std::vector<BlockVector> modes_k_prev; 
			/// Block Vector for all the fractional modes of the modes at the current time 
			std::vector<BlockVector> modes_k; 
			/// Block Vector for all the fractional modes of the modes at the current time 
			std::vector<BlockVector> modes_k_next; 

			std::vector<ParGridFunction> dt_1malpha_modes_gf;

			/// Vector of vector of fractional modes of individual GridFunctions for the next modes
			std::vector<std::vector<ParGridFunction>> modes_k_prev_gf;
			/// Vector of vector of fractional modes of individual GridFunctions for the current modes
			std::vector<std::vector<ParGridFunction>> modes_k_gf;
			/// Vector of vector of fractional modes of individual GridFunctions for the next modes
			std::vector<std::vector<ParGridFunction>> modes_k_next_gf;
			/// GridFunctions for the extrapolated extra-stress 
			Array2D<ParGridFunction *> extra_stress_np2_gf; 


			// weights of the kernel compression 
			Array<double> weights; 
			// poles / exponents of the kernel compression 
			Array<double> poles; 
			// gamma_k coefficients of the time stepping scheme
			Array<double> gamma_k; 
			// delta_k coefficients of the time stepping scheme
			Array<double> delta_k; 
			// beta_1_k coefficients of the time stepping scheme
			Array<double> beta1_k; 
			// beta_2_k coefficients of the time stepping scheme
			Array<double> beta2_k; 
			// weight of the singular term of the kernel compression 
			double w_inf; 
			
			// auxiliary doubles 
			double sum_of_weights=0;
			double sum_of_poles_beta2=0; 
			double sum_of_weights_m_poles_beta1=0; 
		};
	}
}