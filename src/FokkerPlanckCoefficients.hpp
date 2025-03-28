#pragma once

#include "mfem.hpp"

namespace mfem
{
    namespace fokker
    {
        /**
         * @brief The Fokker-Planck coefficients for different potentials.
         *
         * This class contains the coefficients of the FokkerPlanckOperator
         * after application of the 2D spherical harmonics in configuration space.
         * These coefficients are the coupling coefficient to lower modes
         * \f[
         *      C^\phi_k = \frac{\int_0^{\tfrac{1}{2}R^2} \exp\big(-U(s)\big) s^{k-1} \,ds}
         *                      {2 \int_0^{\tfrac{1}{2}R^2} \exp\big(-U(s)\big) s^k \,ds} ,
         * \f]
         * where \f$ U \f$ is the entropic spring potential, and the extra-stress
         * coefficient \f$ C_\tau = (C^\phi_1)^{-1} \f$.
         *
         * @param type The entropic spring potential type
         * @param dimension The dimension to work in (currently only 2D supported)
         */
        class FokkerPlanckCoefficients
        {
        public:
            /// @brief The entropic spring potential type.
            enum class Potential
            {
                LINEAR, ///< Linear elastic potential \f$ U(s) = s \f$
                FENE    ///< Finitely-extensible nonlinear elastic potential \f$ U(s) = -\frac{1}{2} \ln(1 - 2 s) \f$
            };

            FokkerPlanckCoefficients() = delete;

            /// @brief Returns the mode coupling coefficient \f$ k C^\phi_k \f$
            /// @param mode the mode (pair) \f$ k \f$
            /// @param type The Potential type
            /// @param dimension The space dimension
            static constexpr double Coefficient(const int mode, const Potential type, const int dimension = 2)
            {
                MFEM_ASSERT(dimension == 2, "Only dimension 2 implemented.");
                switch (type)
                {
                case Potential::LINEAR:
                    return 0.5;
                    break;

                case Potential::FENE:
                    return (mode + 2.0);
                    break;

                default:
                    std::exit(EXIT_FAILURE);
                    break;
                }
            }

            /// @brief Returns the extra-stress coefficient \f$ C_\tau \f$
            /// @param type The Potential type
            /// @param dimension The space dimension
            static constexpr double ExtraStressCoefficient(const Potential type, const int dimension = 2)
            {
                return 1.0 / Coefficient(1, type, dimension);
            }
        };
    }
}