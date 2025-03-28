#include "FokkerPlanckSettings.hpp"

using namespace mfem;
using namespace fokker;

void FokkerPlanckSettings::setOptions(OptionsParser &args)
{
    args.AddOption(&order, "-fpo", "--fp-order", "Finite element order (polynomial degree) for Fokker-Planck.");
    args.AddOption(&space_diff_coeff, "-sdiff", "--space-diff", "Space diffusion coefficient for Fokker-Planck.");
    args.AddOption(&conf_diff_coeff, "-cdiff", "--conf-diff", "Configuration diffusion coefficient for Fokker-Planck.");
    args.AddOption(&stress_iso_coeff, "-esi", "--extra-stress-isotropic", "Extra-stress isotropic coefficient for Fokker-Planck.");
    args.AddOption(&stress_aniso_coeff, "-esa", "--extra-stress-anisotropic", "Extra-stress anisotropic coefficient for Fokker-Planck.");
    // TODO: improve cast for runtime check
    args.AddOption((int *)&potential, "-fpp", "--fp-potential",
                   "Which entropic spring potential to use in Fokker-Planck (0 = linear, 1 = FENE).");
    args.AddOption((int *)&configuration, "-fpc", "--fp-configuration",
                   "Which discretization to use for the configuration space (0 = spherical harmonics, 1 = tensorial Hermite polynomiasl).");               
    args.AddOption(&max_mode, "-k", "--max-mode", "Maximal mode of the spherical harmonics.");
    args.AddOption(&compute_odd_modes, "-odd", "--odd-modes", "-no-odd", "--no-odd-modes",
                   "Whether to compute odd modes of the spherical harmonics (FP).");
    args.AddOption(&abs_tol, "-fpatol", "--fp-abs-tol", "Solver absolute tolerance for Fokker-Planck.");
    args.AddOption(&rel_tol, "-fprtol", "--fp-rel-tol", "Solver relative tolerance for Fokker-Planck.");
    args.AddOption(&max_iter, "-fpiter", "--fp-max-iter", "Solver maximal iterations for Fokker-Planck.");
    args.AddOption(&verbose, "-fpv", "--fp-verbose", "-fp-nv", "-fp-non-verbose",
                   "Whether to show additional information for Fokker-Planck solver.");
}