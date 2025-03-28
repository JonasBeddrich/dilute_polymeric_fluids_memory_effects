#pragma once

#include <string>

#include "FokkerPlanckSettings.hpp"
#include "TimeIntegrationSettings.hpp"
#include "mfem.hpp"
#include "miniapps/navier/navier_solver.hpp"

namespace mfem {
namespace fokker {
/// Combination of all settings for run cases
class FP_RunSetup {
 public:
  /// The different run cases
  enum class RunCase {
    // 0-5
    ZERO,       ///< \f$ u = (0, 0) \f$ and \f$ kappa = (0, 0; 0, 0) \f$
    CONV_DIAG,  ///< \f$ u = 0 \f$ and \f$ kappa = (1, 0; 0, -1) \f$
    CONV_OFFD,  ///< \f$ u = 0 \f$ and \f$ kappa = (0, 1; 1, 0) \f$
    CONV_FULL,  ///< \f$ u = 0 \f$ and \f$ kappa = (1, 1; 1, -1) \f$
    CO_ROT,     ///< \f$ u = 0 \f$ and \f$ kappa = (0, 1; -1, 0) \f$
    COS,
    // 3D 6-8 
    ZERO3D,        ///< \f$ u = 0 \f$ and \f$ kappa = 0   \f$
    CONV_PDIAG3D,  ///< \f$ u = 0 \f$ and \f$ kappa = Id  \f$
    CONV_NDIAG3D,  ///< \f$ u = 0 \f$ and \f$ kappa = -Id \f$
    // 9-14
    CONV_POFFD3D1, ///< \f$ u = 0 \f$ and \f$ kappa = (0, 1, 0; 0, 0, 0; 0, 0, 0) \f$
    CONV_POFFD3D2, ///< \f$ u = 0 \f$ and \f$ kappa = (0, 0, 1; 0, 0, 0; 0, 0, 0) \f$
    CONV_POFFD3D3, ///< \f$ u = 0 \f$ and \f$ kappa = (0, 0, 0; 1, 0, 0; 0, 0, 0) \f$
    CONV_POFFD3D4, ///< \f$ u = 0 \f$ and \f$ kappa = (0, 0, 0; 0, 0, 1; 0, 0, 0) \f$
    CONV_POFFD3D5, ///< \f$ u = 0 \f$ and \f$ kappa = (0, 0, 0; 0, 0, 0; 1, 0, 0) \f$
    CONV_POFFD3D6, ///< \f$ u = 0 \f$ and \f$ kappa = (0, 0, 0; 0, 0, 0; 0, 1, 0) \f$
    // 15-20
    CONV_NOFFD3D1, ///< \f$ u = 0 \f$ and \f$ kappa = (0, -1,  0;  0, 0,  0;  0,  0, 0) \f$
    CONV_NOFFD3D2, ///< \f$ u = 0 \f$ and \f$ kappa = (0,  0, -1;  0, 0,  0;  0,  0, 0) \f$
    CONV_NOFFD3D3, ///< \f$ u = 0 \f$ and \f$ kappa = (0,  0,  0; -1, 0,  0;  0,  0, 0) \f$
    CONV_NOFFD3D4, ///< \f$ u = 0 \f$ and \f$ kappa = (0,  0,  0;  0, 0, -1;  0,  0, 0) \f$
    CONV_NOFFD3D5, ///< \f$ u = 0 \f$ and \f$ kappa = (0,  0,  0;  0, 0,  0; -1,  0, 0) \f$
    CONV_NOFFD3D6, ///< \f$ u = 0 \f$ and \f$ kappa = (0,  0,  0;  0, 0,  0;  0, -1, 0) \f$    

    CO_ROT3D, ///< \f$ u = 0 \f$ and \f$ kappa = (0,-1,-1; 1,0,-1;1,1,0) \f$
  };

  /// The run case
  int run_case = 0;

  /// The settings for the Fokker-Planck solver
  FokkerPlanckSettings fp_settings;
  /// The settings for the time integration
  TimeIntegrationSettings time_settings;

  /// Returns the mesh
  ParMesh &GetMesh() { return *pmesh; }

  /// Space advection coefficient \f$ u \f$
  VectorCoefficient &getSpaceAdvectionCoeff() { return *space_adv_coeff; }
  /// Configuration advection coefficient \f$ kappa \f$
  MatrixArrayCoefficient &getConfigurationAdvectionCoeff() {
    return *conf_adv_coeff;
  }

  /// Whether the analytic solution is known
  virtual bool has_analytic_solution() {
    switch (fp_settings.configuration) {
      case FokkerPlanckSettings::Configuration::SPHERICAL:
        return run_case <= static_cast<int>(RunCase::CO_ROT);
        break;
      case FokkerPlanckSettings::Configuration::TENSORIAL:
        // For the tensor approach, the CONV_FULL a quite complicated ODE system
        return run_case <= static_cast<int>(RunCase::CONV_OFFD) ||
               run_case == static_cast<int>(RunCase::CO_ROT) ||
               run_case >= static_cast<int>(RunCase::ZERO3D);
        break;
    }
    return false;
  }

  /// Returns the analytic solution for all modes (nullptr if unknown)
  VectorArrayCoefficient *GetAnalyticModes(const double time) {
    if (has_analytic_solution()) {
      analytic_modes->SetTime(time);
      return analytic_modes;
    } else {
      return nullptr;
    }
  };

  /// Adds all options to the Option parser
  virtual void setOptions(OptionsParser &args);

  /// Initialize the setup for set run case.
  /// @todo Extend for 3D cases
  virtual void initialize();

  virtual ~FP_RunSetup();

 protected:
  /// The mesh
  ParMesh *pmesh = nullptr;
  /// The mesh file
  std::string meshfile = "../data/inline-quad.mesh";
  // The serial mesh refinement level for the mesh
  int ser_refine_levels = -1;
  // The parallel mesh refinement level for the mesh
  int par_refine_levels = -1;
  /// Spatial advection coefficient
  VectorCoefficient *space_adv_coeff = nullptr;
  /// Configuration advection coefficient
  MatrixArrayCoefficient *conf_adv_coeff = nullptr;

  ConstantCoefficient *rwa = nullptr; 

  /// Dirichlet BC attributes
  Array<int> dirichlet_bdr;
  /// Dirichlet BC values
  VectorArrayCoefficient *dirichlet_values = nullptr;

  /// Outflow BC attributes
  Array<int> outflow_bdr;

  /// Analytic solution of modes
  VectorArrayCoefficient *analytic_modes = nullptr;

  /// Whether initialize was already called
  bool is_set = false;

  /// Zero vector
  Vector zero_vec;

  /// Initialize and refine the mesh
  void CreateMesh();

  /// Sets the spatial and configuration parameters
  void SetParameters();

  /// Sets the analytical solution
  void SetSolution();
};

/// Combination of all settings for NS-FP run cases
class NSFP_RunSetup : public FP_RunSetup {
 public:
  /// The different run cases
  enum class RunCase {
    CIRCLE,      ///< Circular flow in a square
    CAVITY,      ///< Lid-driven cavity
    KARMAN,      ///< Karman vortex street
    DONUT,       ///< Donut mesh
    KARMAN_DFG,  /// flow over cylinder DFG benchmark 
    RW_CHANNEL,  /// flow in rough wall channel 
    LDC_3D, 
    PSR_3D, 
    POISEUILLE2D
  };

  /// FEM order (polynomial degree) of NavierSolver
  int ns_order = 2;
  /// Kinematic viscosity coefficient of NavierSolver
  double ns_kin_viscosity = 1.0;
  /// Whether to use partial assembly for NavierSolver
  bool ns_pa = false;
  /// Whether to use numerical integration for NavierSolver
  bool ns_ni = false;
  /// Whether to output info for NavierSolver
  bool ns_verbose = false;

  /// Navier-Stokes velocity BC attributes
  Array<int> velocity_bdr;
  /// Navier-Stokes velocity BC value
  VectorCoefficient *velocity_bdr_value = nullptr;

  /// Navier-Stokes pressure BC attributes
  Array<int> pressure_bdr;
  /// Navier-Stokes velocity BC value
  Coefficient *pressure_bdr_value = nullptr;

  /// Navier-Stokes acceleration domain attributes
  Array<int> acceleration_dom;
  /// Navier-Stokes acceleration value
  VectorCoefficient *acceleration_value = nullptr;

  void CreateMesh(); 

  /// Whether the analytic solution is known
  virtual bool has_analytic_solution() { return false; }

  /// Adds all options to the Option parser
  virtual void setOptions(OptionsParser &args) override;

  /// Initialize the setup for set run case.
  /// @todo Extend for 3D cases
  virtual void initialize() override;

  virtual ~NSFP_RunSetup() {}

 protected:
};

class NavierSolver2 : public navier::NavierSolver {
 public:
  /** @brief Initialize data structures from @a setup.
   *
   * The ParMesh @a mesh can be a linear or curved parallel mesh. The @setup
   * controls the order of the finite element spaces, the kinematic viscosity
   * (dimensionless) and further numerical settings.
   */
  NavierSolver2(ParMesh *mesh, NSFP_RunSetup &setup);
};
}  // namespace fokker
}  // namespace mfem