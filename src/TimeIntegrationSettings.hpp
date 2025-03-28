#pragma once

#include "mfem.hpp"

namespace mfem
{
    /// This class contains all parameters concerning the FokkerPlanckOperator
    class TimeIntegrationSettings
    {
    public:
        /// Type of time integrator
        enum class TimeIntegratorType : unsigned
        {
            ForwardEuler,     ///< Explicit Euler
            BackwardEuler,    ///< Implicit Euler
            ImplicitMidpoint, ///< Implicit midpoint rule
            RK3SSP,           ///< Strong stability preserving Runge-Kutta of order 3
            TrapezoidalRule,  ///< Trapezoidal rule
        };

        /// Final simulation time
        double t_final = 1.0; // Final simulation time
        /// Time-step size
        double dt = 1.0e-2; // Time-step size
        /// Time integrator type
        int ode_solver_type = 1;

        /// @brief Adds all options to the Option parser
        /// @todo: Add options for PotentialType and print levels
        void setOptions(OptionsParser &args);

        /// Returns the ODESolver chosen in options (releases ownership)
        ODESolver *getTimeIntegrator() const;
    };
}