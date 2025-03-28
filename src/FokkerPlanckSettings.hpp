#pragma once

#include "mfem.hpp"
#include "FokkerPlanckCoefficients.hpp"

namespace mfem
{
    namespace fokker
    {
        /**
         * @class FokkerPlanckSettings
         * @brief This class contains all parameters concerning the FokkerPlanckOperator
         */
        class FokkerPlanckSettings
        {
        public:

        	/// @brief The discretization of the configuration space  
			enum class Configuration
            {
                SPHERICAL,	/// 0 - using spherical harmonics                 
				TENSORIAL   /// 1 - using a tensor product of Hermite polynomials 
            };

            /// FEM order (polynomial degree)
            int order = 2;

            /// Spatial diffusion coefficient
            double space_diff_coeff = 1.0;
            /// Configuration diffusion coefficient // chi or 1/De 
            double conf_diff_coeff = 1.0;

            /// Extra-stress isotropic coefficient
            double stress_iso_coeff = 1.0;
            /// Extra-stress anisotropic coefficient
            double stress_aniso_coeff = 1.0;

            /// The entropic spring potential
            FokkerPlanckCoefficients::Potential potential = FokkerPlanckCoefficients::Potential::FENE;

            /// The discretization of the configuration space 
            FokkerPlanckSettings::Configuration configuration = FokkerPlanckSettings::Configuration::TENSORIAL; 

            /// width scaling of the hermite polynomials - this is "a" 
            // double hermite_scaling = 1. / 2.; 
            double hermite_scaling = 1. / sqrt(2.); 
            

            /// Maximal degree of (spherical harmonics) modes
            int max_mode = 2;
            /// Whether to compute odd modes (asymmetric)
            bool compute_odd_modes = false;

            /// Whether to show additional information
            bool verbose = false;

            /// Solver absolute tolerance
            double abs_tol = 0.0;
            /// Solver relative tolerance
            double rel_tol = 1e-12;
            /// Solver maximal iterations
            int max_iter = 100;

            /// Print level of the mass solver
            IterativeSolver::PrintLevel print_level_M;
            /// Print level of the advection-diffusion-reaction solver
            IterativeSolver::PrintLevel print_level_T;
            /// Print level of the block-system solver
            IterativeSolver::PrintLevel print_level_fp;

            /// @brief Adds all options to the Option parser
            /// @todo: Add options for PotentialType and print levels
            void setOptions(OptionsParser &args);

            /// Returns the total number of modes
            inline int GetNModes(const int dimension=0) const
            {   
                switch(configuration)
                {
                    case Configuration::SPHERICAL: 
                        MFEM_ASSERT(dimension == 2, "Only dimension 2 is allowed.")
                        return compute_odd_modes ? 1 + 2 * max_mode : 1 + 2 * GetNEven();
                        break; 

                    case Configuration::TENSORIAL: 
                        MFEM_ASSERT(dimension == 2 || dimension == 3, "Only dimensions 2 and 3 are allowed.")
                        if (dimension == 2){
                            return 4; 
                        } else if (dimension == 3){
                            return 7; 
                        }
                        break; 
                }
                return 0; 
            }

            /// Returns the number of even mode pairs
            inline int GetNEven() const
            {
                return max_mode / 2; // NOTE: integer division intended!
            }

            /// Returns the number of odd mode pairs
            inline int GetNOdd() const
            {
                return compute_odd_modes ? max_mode - GetNEven() : 0;
            }

            /// @brief Returns the number of mode pairs
            ///
            /// This means `max_mode` if `compute_odd_modes` is `true`,
            /// `GetNEven(max_mode)` otherwise
            inline int GetNPairs() const
            {
                return compute_odd_modes ? max_mode : GetNEven();
            }

            // Computes the mode number for given vector index (pair)
            inline int IndexToMode(const int index) const
            {
                return (index <= GetNEven()) ? 2 * index : 2 * (index - GetNEven()) - 1;
            }

            // Computes the vector index (first of pair) for given mode number
            inline int ModeToIndex(const int mode) const
            {
                return (mode % 2 == 0) ? mode / 2 : GetNEven() + (mode + 1) / 2;
            } 
        };
    }
}