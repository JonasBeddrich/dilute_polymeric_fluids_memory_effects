#define MFEM_USE_MPI

#include "FokkerPlanckSolver.hpp"

using namespace mfem;
using namespace fokker;

FokkerPlanckSolver::FokkerPlanckSolver(ParMesh &mesh,
                                       FokkerPlanckSettings &settings)
    : settings(settings),
      pmesh(mesh),
      fec(settings.order, pmesh.Dimension()),
      fes(&pmesh, &fec),
      offsets(settings.GetNModes(pmesh.Dimension()) + 1),
      modes_gf(settings.GetNModes(pmesh.Dimension())),
      modes_next_gf(settings.GetNModes(pmesh.Dimension())),
      extra_stress_gf(pmesh.Dimension(), pmesh.Dimension()),
      extra_stress_next_gf(pmesh.Dimension(), pmesh.Dimension()) {
  // Create offset vectors for modes
  offsets = fes.GetTrueVSize();
  offsets[0] = 0;
  offsets.PartialSum();

  // block vectors
  modes.Update(offsets);
  modes_next.Update(offsets);

  ConstantCoefficient init_coeff;

  // initialize grid functions of other modes
  switch (settings.configuration) {
    case (FokkerPlanckSettings::Configuration::SPHERICAL):
      // initialize grid functions of zeroth mode (=1)
      modes = 0.0;
      modes_gf[0].MakeTRef(&fes, modes, offsets[0]);
      init_coeff = ConstantCoefficient(1.0);
      modes_gf[0].ProjectCoefficient(init_coeff);
      modes_gf[0].SetTrueVector();
      modes_gf[0].SetFromTrueVector();

      modes_next = modes;
      modes_next_gf[0].MakeTRef(&fes, modes_next, offsets[0]);
      modes_next_gf[0].SetFromTrueVector();

      for (int k = 1; k <= settings.GetNPairs(); ++k) {
        modes_gf[2 * k - 1].MakeTRef(&fes, modes, offsets[2 * k - 1]);
        modes_gf[2 * k].MakeTRef(&fes, modes, offsets[2 * k]);
        modes_gf[2 * k - 1].SetFromTrueVector();
        modes_gf[2 * k].SetFromTrueVector();

        modes_next_gf[2 * k - 1].MakeTRef(&fes, modes_next, offsets[2 * k - 1]);
        modes_next_gf[2 * k].MakeTRef(&fes, modes_next, offsets[2 * k]);
        modes_next_gf[2 * k - 1].SetFromTrueVector();
        modes_next_gf[2 * k].SetFromTrueVector();
      }
      break;

    case (FokkerPlanckSettings::Configuration::TENSORIAL):
      modes = 0.0;
      for (int k = 0; k < settings.GetNModes(pmesh.Dimension()); ++k) {
        modes_gf[k].MakeTRef(&fes, modes, offsets[k]);
      }

      // initial condition for 0 mode
      init_coeff = ConstantCoefficient(1);
      modes_gf[0].ProjectCoefficient(init_coeff);

      switch (pmesh.Dimension()) {
        case (2):
          // initial condition for 11 mode 
          init_coeff = ConstantCoefficient(0.0);
          modes_gf[1].ProjectCoefficient(init_coeff);
          // initial conditions for 02 modes 
          init_coeff = ConstantCoefficient(
              (2 * std::pow(settings.hermite_scaling, 2) - 1) / sqrt(2));
          modes_gf[2].ProjectCoefficient(init_coeff);
          modes_gf[3].ProjectCoefficient(init_coeff);
          break;
        case (3):
          // initial conditions for 110 modes
          init_coeff = ConstantCoefficient(0.0);
          modes_gf[1].ProjectCoefficient(init_coeff);
          modes_gf[2].ProjectCoefficient(init_coeff);
          modes_gf[3].ProjectCoefficient(init_coeff);
          // initial conditions for 002 modes
          init_coeff = ConstantCoefficient(
              (2 * std::pow(settings.hermite_scaling, 2) - 1) / sqrt(2));
          modes_gf[4].ProjectCoefficient(init_coeff);
          modes_gf[5].ProjectCoefficient(init_coeff);
          modes_gf[6].ProjectCoefficient(init_coeff);
          break;
        default:
          MFEM_ABORT("Only dimensions 2 and 3 are allowed.")
          break;
      }

      for (int k = 0; k < settings.GetNModes(pmesh.Dimension()); ++k) {
        modes_gf[k].SetTrueVector();
        modes_gf[k].SetFromTrueVector();
      }
      modes_next = modes;
      for (int k = 0; k < settings.GetNModes(pmesh.Dimension()); ++k) {
        modes_next_gf[k].MakeTRef(&fes, modes_next, offsets[k]);
        modes_next_gf[k].SetFromTrueVector();
      }
      break;
  }

  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      extra_stress_gf[i][j] = new ParGridFunction(&fes);
      extra_stress_next_gf[i][j] = new ParGridFunction(&fes);
    }
  }

  ComputeExtraStress();

  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      *extra_stress_gf[i][j] = *extra_stress_next_gf[i][j];
    }
  }

  // Check for fully periodic mesh
  if (!(pmesh.bdr_attributes.Size() == 0)) {
    dirichlet_attr.SetSize(pmesh.bdr_attributes.Max());
    dirichlet_attr = 0;

    outflow_attr.SetSize(pmesh.bdr_attributes.Max());
    outflow_attr = 0;
  }

  PrintInfo();
}

void FokkerPlanckSolver::Setup(ODESolver *solver) {
  if (settings.verbose && Mpi::Root()) {
    out << "Fokker-Planck Setup" << std::endl;
  }

  sw_setup.Start();

  if (!fp_operator) {
    fes.GetEssentialTrueDofs(dirichlet_attr, dirichlet_tdof);
    switch (settings.configuration) {
      case (FokkerPlanckSettings::Configuration::SPHERICAL):
        fp_operator = new FokkerPlanckOperator(
            fes, dirichlet_tdof, spatial_adv_terms[0].coeff,
            configuration_adv_terms[0].coeff, settings);
        break;

      case (FokkerPlanckSettings::Configuration::TENSORIAL):
        fp_operator = new TensorFokkerPlanckOperator(
            fes, dirichlet_tdof, spatial_adv_terms[0].coeff,
            configuration_adv_terms[0].coeff, settings);
        break;
    }
  } else {
    fp_operator->SetParameters(spatial_adv_terms[0].coeff,
                               configuration_adv_terms[0].coeff);
  }

  if (solver) {
    delete ode_solver;
    ode_solver = solver;
  }
  ode_solver->Init(*fp_operator);
  sw_setup.Stop();
}

void FokkerPlanckSolver::Step(double &time, double dt, int current_step,
                              bool provisional) {
  sw_step.Start();

  // Set current time for boundary conditions and advection terms.
  for (auto &bc : dirichlet_bcs) {
    bc.coeff.SetTime(time + dt);
    for (int i = 0; i < bc.coeff.GetVDim(); ++i) {
      modes_next_gf[i].ProjectBdrCoefficient(*bc.coeff.GetCoeff(i), bc.attr);
      modes_next_gf[i].SetTrueVector();
    }
  }

  for (auto &term : spatial_adv_terms) {
    term.coeff.SetTime(time + dt);
  }

  for (auto &term : configuration_adv_terms) {
    term.coeff.SetTime(time + dt);
  }

  fp_operator->SetParameters(spatial_adv_terms[0].coeff,
                             configuration_adv_terms[0].coeff);

  ode_solver->Step(modes_next, time, dt);

  for (auto &gf : modes_next_gf) {
    gf.SetFromTrueVector();
  }

  // Update extra-stress
  ComputeExtraStress();

  // If the current time step is not provisional, accept the computed solution
  // and update the time step history by default.
  if (provisional) {
    time -= dt;
  } else {
    UpdateTimestepHistory(dt);
  }

  sw_step.Stop();
}

void FokkerPlanckSolver::AddDirichletBC(Array<int> &attr,
                                        VectorArrayCoefficient &coeff) {
  dirichlet_bcs.emplace_back(attr, coeff);

  if (settings.verbose && Mpi::Root()) {
    out << "Adding Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        out << i << " ";
      }
    }
    out << std::endl;
  }

  MFEM_ASSERT(coeff.Size() == settings.GetNModes(),
              "Size of Dirichlet BCs does not match numer of modes.");
  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((dirichlet_attr[i] && attr[i]) == 0,
                "Duplicate boundary definition detected.");
    if (attr[i] == 1) {
      dirichlet_attr[i] = 1;
    }
  }
}

void FokkerPlanckSolver::AddOutflowBC(const Array<int> &attr) {
  if (settings.verbose && Mpi::Root()) {
    out << "Adding outflow BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        out << i << " ";
      }
    }
    out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      outflow_attr[i] = 1;
    }
  }
}

void FokkerPlanckSolver::AddSpaceAdvectionTerm(Array<int> &attr,
                                               VectorCoefficient &coeff) {
  spatial_adv_terms.emplace_back(attr, coeff);

  if (settings.verbose && Mpi::Root()) {
    out << "Adding space advection term to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    out << std::endl;
  }
}

void FokkerPlanckSolver::AddConfigurationAdvectionTerm(
    Array<int> &attr, MatrixArrayCoefficient &coeff) {
  configuration_adv_terms.emplace_back(attr, coeff);

  if (settings.verbose && Mpi::Root()) {
    mfem::out << "Adding configuration advection term to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }
}

void FokkerPlanckSolver::ComputeExtraStress() {
  switch (settings.configuration) {
    case (FokkerPlanckSettings::Configuration::SPHERICAL):
      for (int i = 0; i < pmesh.Dimension(); ++i) {
        for (int j = 0; j < pmesh.Dimension(); ++j) {
          const double coeff = settings.stress_aniso_coeff *
                               FokkerPlanckCoefficients::ExtraStressCoefficient(
                                   settings.potential);
          if (i == j) {
            extra_stress_next_gf[i][j]->Set(settings.stress_iso_coeff,
                                            modes_next_gf[0]);
            extra_stress_next_gf[i][j]->Add(i ? -coeff : coeff,
                                            modes_next_gf[1]);
          } else {
            extra_stress_next_gf[i][j]->Set(coeff, modes_next_gf[2]);
          }
        }
      }
      break;

    case (FokkerPlanckSettings::Configuration::TENSORIAL):
      double a2 = std::pow(settings.hermite_scaling, 2.);
      double g = settings.stress_iso_coeff;

      switch (pmesh.Dimension()) {
        case (2): {
          // phi00 
          extra_stress_next_gf[0][0]->Set(g * (1 / 2 / a2 - 1), modes_next_gf[0]);  
          extra_stress_next_gf[1][1]->Set(g * (1 / 2 / a2 - 1), modes_next_gf[0]);  

          // phi11 
          extra_stress_next_gf[0][1]->Set(g / 2 / a2, modes_next_gf[1]);  
          extra_stress_next_gf[1][0]->Set(g / 2 / a2, modes_next_gf[1]);  

          // phi02 phi20 
          extra_stress_next_gf[0][0]->Set(g / sqrt(2) / a2, modes_next_gf[3]); 
          extra_stress_next_gf[1][1]->Set(g / sqrt(2) / a2, modes_next_gf[2]); 
          break;
        }
        case (3): {
          // phi000
          extra_stress_next_gf[0][0]->Set(g * (1 / 2 / a2 - 1), modes_next_gf[0]);  
          extra_stress_next_gf[1][1]->Set(g * (1 / 2 / a2 - 1), modes_next_gf[0]);  
          extra_stress_next_gf[2][2]->Set(g * (1 / 2 / a2 - 1), modes_next_gf[0]);  

          // phi011 phi101 phi110
          extra_stress_next_gf[0][1]->Set(g / 2 / a2, modes_next_gf[3]);  
          extra_stress_next_gf[0][2]->Set(g / 2 / a2, modes_next_gf[2]);  
          extra_stress_next_gf[1][0]->Set(g / 2 / a2, modes_next_gf[3]);  
          extra_stress_next_gf[1][2]->Set(g / 2 / a2, modes_next_gf[1]);  
          extra_stress_next_gf[2][0]->Set(g / 2 / a2, modes_next_gf[2]);  
          extra_stress_next_gf[2][1]->Set(g / 2 / a2, modes_next_gf[1]);  

          // phi002 phi020 phi200
          extra_stress_next_gf[0][0]->Set(g / sqrt(2) / a2, modes_next_gf[6]);
          extra_stress_next_gf[1][1]->Set(g / sqrt(2) / a2, modes_next_gf[5]);
          extra_stress_next_gf[2][2]->Set(g / sqrt(2) / a2, modes_next_gf[4]);  
          break;
        }
        default:
          MFEM_ABORT("Only dimensions 2 and 3 are allowed.")
          break;
      }
      break;
  }
}

void FokkerPlanckSolver::UpdateTimestepHistory(double dt) {
  // Update the current solution and corresponding GridFunctions
  modes = modes_next;
  for (auto &gf : modes_gf) {
    gf.SetFromTrueVector();
  }
  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      *extra_stress_gf[i][j] = *extra_stress_next_gf[i][j];
    }
  }
}

void FokkerPlanckSolver::AddModesToDataCollection(DataCollection &dc) {
  dc.RegisterField("phi0", &modes_gf[0]);

  switch (settings.configuration) {
    case (FokkerPlanckSettings::Configuration::SPHERICAL):
      for (int k = 1; k <= settings.GetNPairs(); ++k) {
        auto name = "phi" + std::to_string(settings.IndexToMode(k));
        dc.RegisterField(name + 'c', &modes_gf[2 * k - 1]);
        dc.RegisterField(name + 's', &modes_gf[2 * k]);
      }
      break;

    case (FokkerPlanckSettings::Configuration::TENSORIAL):
      switch (pmesh.Dimension()) {
        case (2): {
          dc.RegisterField("phi11", &modes_gf[1]);
          dc.RegisterField("phi02", &modes_gf[2]);
          dc.RegisterField("phi20", &modes_gf[3]);
          break;
        }
        case (3): {
          dc.RegisterField("phi011", &modes_gf[1]);
          dc.RegisterField("phi101", &modes_gf[2]);
          dc.RegisterField("phi110", &modes_gf[3]);
          dc.RegisterField("phi002", &modes_gf[4]);
          dc.RegisterField("phi020", &modes_gf[5]);
          dc.RegisterField("phi200", &modes_gf[6]);
          break;
        }
        default:
          MFEM_ABORT("Only 2 and 3 dimensions are allowed.")
          break;
      }
      break;
  }
}

void FokkerPlanckSolver::PrintInfo() {
  if (Mpi::Root()) {
    mfem::out << "Fokker-Planck version: " << FOKKER_PLANCK_VERSION << std::endl
              << "MFEM version: " << MFEM_VERSION << std::endl
              << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
              << "Phi #DOFs: " << fes.GlobalVSize() << std::endl
              << "Total Phi #DOFs: "
              << (fes.GlobalVSize() * settings.GetNModes(pmesh.Dimension()))
              << std::endl;
  }
}

void FokkerPlanckSolver::PrintTimingData() {
  double my_rt[2], rt_max[2];

  my_rt[0] = sw_setup.RealTime();
  my_rt[1] = sw_step.RealTime();

  MPI_Reduce(my_rt, rt_max, 2, MPI_DOUBLE, MPI_MAX, 0, pmesh.GetComm());

  if (Mpi::Root()) {
    out << std::fixed << std::setprecision(3) << std::endl
        << "Times: Setup " << my_rt[0] << ", Step " << my_rt[1] << std::endl;
    out << std::defaultfloat;
  }
}

FokkerPlanckSolver::~FokkerPlanckSolver() {
  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      delete extra_stress_gf[i][j];
      delete extra_stress_next_gf[i][j];
    }
  }
  delete fp_operator;
  delete ode_solver;
}