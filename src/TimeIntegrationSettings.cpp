#include <string>
#include "TimeIntegrationSettings.hpp"

using namespace mfem;

void TimeIntegrationSettings::setOptions(OptionsParser &args)
{
    args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step", "Time-step size.");
    args.AddOption(&ode_solver_type, "-s", "--solver", "ODE Solver type (0 = EE, 1 = IE, 2 = MID, 3 = RK3SSP, 4 = TRAP).");
}

ODESolver *TimeIntegrationSettings::getTimeIntegrator() const
{
    switch (ode_solver_type)
    {
    case static_cast<int>(TimeIntegratorType::ForwardEuler):
        return new ForwardEulerSolver;
    case static_cast<int>(TimeIntegratorType::BackwardEuler):
        return new BackwardEulerSolver;
    case static_cast<int>(TimeIntegratorType::ImplicitMidpoint):
        return new ImplicitMidpointSolver;
    case static_cast<int>(TimeIntegratorType::RK3SSP):
        return new RK3SSPSolver;
    case static_cast<int>(TimeIntegratorType::TrapezoidalRule):
        return new TrapezoidalRuleSolver;
    default:
        MFEM_ABORT("Unknown ODE solver type: " + std::to_string(ode_solver_type));
        return nullptr;
    }
}
