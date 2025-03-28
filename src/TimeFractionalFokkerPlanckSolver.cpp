#define MFEM_USE_MPI

#include "TimeFractionalFokkerPlanckSolver.hpp"

using namespace mfem;
using namespace fokker;

TimeFractionalFokkerPlanckSolver::TimeFractionalFokkerPlanckSolver(ParMesh &mesh,
                                       FokkerPlanckSettings &settings,
                                       double alpha_)
    : FokkerPlanckSolver(mesh, settings), 
    alpha(alpha_), 
    modes_k_prev(m),
    modes_k(m),
    modes_k_next(m),  
    modes_k_prev_gf(m, std::vector<ParGridFunction>(settings.GetNModes(pmesh.Dimension()))), 
    modes_k_gf(m, std::vector<ParGridFunction>(settings.GetNModes(pmesh.Dimension()))), 
    modes_k_next_gf(m, std::vector<ParGridFunction>(settings.GetNModes(pmesh.Dimension()))), 
    dt_1malpha_modes_gf(settings.GetNModes(pmesh.Dimension())),
    extra_stress_np2_gf(pmesh.Dimension(), pmesh.Dimension()),
    weights(m), 
    poles(m), 
    gamma_k(m), 
    delta_k(m), 
    beta1_k(m), 
    beta2_k(m)
    {

      inFn.Update(offsets); 
      inFn1.Update(offsets);

      modes_prev.Update(offsets); 
      modes_prev.Set(1.,modes); 

      dt_1malpha_modes.Update(offsets); 
      dt_1malpha_modes = 0.; 

      for (int i = 0; i < dt_1malpha_modes_gf.size(); i++){ 
        dt_1malpha_modes_gf[i].MakeTRef(&fes, dt_1malpha_modes, offsets[i]);
        dt_1malpha_modes_gf[i].SetFromTrueVector(); 
      }

      tmp.Update(offsets); 
      tmp2.Update(offsets); 
      rhs.Update(offsets); 

      for (int k = 0; k < m; ++k){
        modes_k_prev[k].Update(offsets); 
        modes_k_prev[k] = 0.;

        modes_k[k].Update(offsets); 
        modes_k[k] = 0.;

        modes_k_next[k].Update(offsets); 
        modes_k_next[k] = 0.;

        for (int i = 0; i < modes_k_gf[k].size(); i++){
          modes_k_prev_gf[k][i].MakeTRef(&fes, modes_k_prev[k], offsets[i]);
          modes_k_prev_gf[k][i].SetFromTrueVector(); 

          modes_k_gf[k][i].MakeTRef(&fes, modes_k[k], offsets[i]);
          modes_k_gf[k][i].SetFromTrueVector(); 

          modes_k_next_gf[k][i].MakeTRef(&fes, modes_k_next[k], offsets[i]);
          modes_k_next_gf[k][i].SetFromTrueVector(); 
        }
      }  // if(first_iteration){
  //   first_iteration = false; 
  //   for (int i = 0; i < settings.GetNModes(dim) * (settings.GetNModes(dim) - 1); ++i) {
  //     Cmat_old[i] = Add(0., *Cmat[i], 1., *Cmat[i]); 
  //   }
  // }


      for (int i = 0; i < pmesh.Dimension(); ++i) {
        for (int j = 0; j < pmesh.Dimension(); ++j) {
          extra_stress_np2_gf[i][j] = new ParGridFunction(&fes);
        }
      }

      ExtrapolateExtraStress(); 
      
      SetWeightsPoles(); 

      for (int k = 0; k < m; ++k){
        sum_of_weights += weights[k]; 
      }
    }

void TimeFractionalFokkerPlanckSolver::Setup(){
  if (settings.verbose && Mpi::Root()) {
    out << "Time-fractional Fokker-Planck Setup" << std::endl;
  }

  sw_setup.Start();

  if (!fp_operator) {
    // set Dirichlet dofs
    fes.GetEssentialTrueDofs(dirichlet_attr, dirichlet_tdof);
    fp_operator = new TensorFokkerPlanckOperator(
      fes, dirichlet_tdof, spatial_adv_terms[0].coeff, 
      configuration_adv_terms[0].coeff, settings);
  } else {
    fp_operator->SetParameters(spatial_adv_terms[0].coeff,
                               configuration_adv_terms[0].coeff);
  }

  sw_setup.Stop();
}

void TimeFractionalFokkerPlanckSolver::Step(double & time, double dt, int current_step, bool provisional){

  sw_step.Start();

  SetModeEquationParameters(dt);

  //****************************************************************
  // Update operator: F^n -> F^n+1
  for (auto &term : spatial_adv_terms) {
    term.coeff.SetTime(time + dt);
  }
  for (auto &term : configuration_adv_terms) {
    term.coeff.SetTime(time + dt);
  }
  fp_operator->SetParameters(spatial_adv_terms[0].coeff,
                             configuration_adv_terms[0].coeff);
  
  //****************************************************************
  // Set current time for boundary conditions and advection terms.
  
  for (auto &bc : dirichlet_bcs) {
    bc.coeff.SetTime(time + dt);
    for (int i = 0; i < bc.coeff.GetVDim(); ++i) {
      modes_next_gf[i].ProjectBdrCoefficient(*bc.coeff.GetCoeff(i), bc.attr);
      modes_next_gf[i].SetTrueVector();
    }
  }

  
  if(alpha == 1.0){
    tmp.Set(4./3., modes); 
    tmp.Add(-1./3., modes_prev);
    inFn1.Set(0., modes); 
    fp_operator->GeneralizedImplicitSolve(
      2. / 3. * dt, 
      tmp, inFn1, modes_next);
  } else{
    if(non_zero_initial){
      //****************************************************************  
      // SDIRK2 for first step  
      const double SDIRK_alpha = (1. - 1. / sqrt(2.)); 

      // Inbetween step
      tmp.Set(1., modes);
      inFn1.Set(0.,modes); 
      double SDIRKdt = 0.; 
      
      for (int k = 0; k < m; ++k){
        // modes are initialized as zero
        // inFn1.Add(- poles[k] * SDIRK_alpha * dt / (1 + poles[k] * SDIRK_alpha * dt), modes_k[k]); 
        SDIRKdt += - poles[k] * weights[k] * SDIRK_alpha * SDIRK_alpha * dt * dt / (1 + SDIRK_alpha * dt * poles[k]);  
        SDIRKdt += weights[k] * SDIRK_alpha * dt; 
      }
      
      // Solve inbetween step 
      fp_operator->GeneralizedImplicitSolve(
        SDIRKdt, 
        tmp, inFn1, modes_next);

      // Inbetween mode update 
      for (int k = 0; k < m; ++k) {
        // as modes are initialized as zero 
        modes_k_next[k].Set(weights[k] * SDIRK_alpha * dt / (1 + SDIRK_alpha * dt * poles[k]) , modes_next); 
      }

      // Closing step 
      tmp.Set(1., modes);
      inFn1.Set(0., modes); 
      SDIRKdt = 0.; 

      for (int k = 0; k < m; ++k){
        inFn1.Add(- (1-SDIRK_alpha) * dt * poles[k], modes_k_next[k]); 
        inFn1.Add((1-SDIRK_alpha) * dt * weights[k], modes_next);
        // modes are initialized as zero 
        // inFn1.Add(- SDIRK_alpha * dt * poles[k] / (1 + SDIRK_alpha * dt * poles[k]), modes_k[k]);
        inFn1.Add(SDIRK_alpha * (1-SDIRK_alpha) * dt * dt * poles[k] * poles[k] / (1 + SDIRK_alpha * dt * poles[k]), modes_k_next[k]);
        inFn1.Add(- SDIRK_alpha * (1-SDIRK_alpha) * dt * dt * weights[k] * poles[k] / (1 + SDIRK_alpha * dt * poles[k]), modes_next);
        SDIRKdt += - SDIRK_alpha * SDIRK_alpha * dt * dt * poles[k] * weights[k] / (1 + SDIRK_alpha * dt * poles[k]); 
        SDIRKdt += SDIRK_alpha * dt * weights[k]; 
      }

      // Closing mode update - part 1 
      for (int k = 0; k < m; ++k) {
        modes_k_next[k].Set(- poles[k] * (1-SDIRK_alpha) * dt / (1 + SDIRK_alpha * dt * poles[k]) , modes_k_next[k]); 
        modes_k_next[k].Add(weights[k] * (1-SDIRK_alpha) * dt / (1 + SDIRK_alpha * dt * poles[k]) , modes_next); 
      }

      // Solve closing step 
      fp_operator->GeneralizedImplicitSolve(
        SDIRKdt, 
        tmp, inFn1, modes_next);

      // Closing mode update - part 2
      for (int k = 0; k < m; ++k) {
        modes_k_next[k].Add( weights[k] * SDIRK_alpha * dt /  // if(first_iteration){
  //   first_iteration = false; 
  //   for (int i = 0; i < settings.GetNModes(dim) * (settings.GetNModes(dim) - 1); ++i) {
  //     Cmat_old[i] = Add(0., *Cmat[i], 1., *Cmat[i]); 
  //   }
  // }
 (1 + SDIRK_alpha * dt * poles[k]) , modes_next); 
      }
    }

    //****************************************************************  
    // Fractional BDF2  
    else{
      tmp.Set(4./3., modes); 
      tmp.Add(-1./3., modes_prev);
      
      inFn1.Set(0., modes); 
      for (int k = 0; k < m; ++k){
        inFn1.Add(- 2. / 3. * dt * poles[k] * gamma_k[k], modes_k[k]);
        inFn1.Add(- 2. / 3. * dt * poles[k] * delta_k[k], modes_k_prev[k]); 
      }
      fp_operator->GeneralizedImplicitSolve(
        2. / 3. * dt * sum_of_weights_m_poles_beta1, 
        tmp, inFn1, modes_next);
    }
  }

  time += dt; 

  UpdateTimestepHistory(dt);

  UpdateTimefractionalDerivative(dt); 
  
  ComputeExtraStress();

  ExtrapolateExtraStress();   // if(first_iteration){
  //   first_iteration = false; 
  //   for (int i = 0; i < settings.GetNModes(dim) * (settings.GetNModes(dim) - 1); ++i) {
  //     Cmat_old[i] = Add(0., *Cmat[i], 1., *Cmat[i]); 
  //   }
  // }


  for (auto &gf : modes_next_gf) {
    gf.SetFromTrueVector();
  }

  non_zero_initial = false; 
  sw_step.Stop();
}

void TimeFractionalFokkerPlanckSolver::UpdateTimefractionalDerivative(const double dt) {
  dt_1malpha_modes = 0.; 
  // This replaces dt^{1-\alpha} \psi in the definition of the coupling tensor with \psi 
  dt_1malpha_modes.Set(1,modes_next); 

  for (auto &gf : dt_1malpha_modes_gf) {
    gf.SetFromTrueVector();
  }
}

void TimeFractionalFokkerPlanckSolver::ComputeExtraStress() {

  double a2 = std::pow(settings.hermite_scaling, 2.);
  double g = settings.stress_iso_coeff;

  switch (pmesh.Dimension()) {
    case (2): {
      // phi00 
      // TODO: This is only correct as 1/2/a2 -1 = 0
      extra_stress_next_gf[0][0]->Set(g * (1 / 2 / a2 - 1), dt_1malpha_modes_gf[0]);  
      extra_stress_next_gf[1][1]->Set(g * (1 / 2 / a2 - 1), dt_1malpha_modes_gf[0]);  

      // phi11 
      extra_stress_next_gf[0][1]->Set(g / 2 / a2, dt_1malpha_modes_gf[1]);  
      extra_stress_next_gf[1][0]->Set(g / 2 / a2, dt_1malpha_modes_gf[1]);  

      // phi02 phi20 
      extra_stress_next_gf[0][0]->Set(g / sqrt(2) / a2, dt_1malpha_modes_gf[3]); 
      extra_stress_next_gf[1][1]->Set(g / sqrt(2) / a2, dt_1malpha_modes_gf[2]); 
      break;
    }
    case (3): {
      // phi000
      // TODO: This is only correct as 1/2/a2 -1 = 0
      extra_stress_next_gf[0][0]->Set(g * (1 / 2 / a2 - 1), dt_1malpha_modes_gf[0]);  
      extra_stress_next_gf[1][1]->Set(g * (1 / 2 / a2 - 1), dt_1malpha_modes_gf[0]);  
      extra_stress_next_gf[2][2]->Set(g * (1 / 2 / a2 - 1), dt_1malpha_modes_gf[0]);  

      // phi011 phi101 phi110
      extra_stress_next_gf[0][1]->Set(g / 2 / a2, dt_1malpha_modes_gf[3]);  
      extra_stress_next_gf[0][2]->Set(g / 2 / a2, dt_1malpha_modes_gf[2]);  
      extra_stress_next_gf[1][0]->Set(g / 2 / a2, dt_1malpha_modes_gf[3]);  
      extra_stress_next_gf[1][2]->Set(g / 2 / a2, dt_1malpha_modes_gf[1]);  
      extra_stress_next_gf[2][0]->Set(g / 2 / a2, dt_1malpha_modes_gf[2]);  
      extra_stress_next_gf[2][1]->Set(g / 2 / a2, dt_1malpha_modes_gf[1]);  

      // phi002 phi020 phi200
      extra_stress_next_gf[0][0]->Set(g / sqrt(2) / a2, dt_1malpha_modes_gf[6]);
      extra_stress_next_gf[1][1]->Set(g / sqrt(2) / a2, dt_1malpha_modes_gf[5]);
      extra_stress_next_gf[2][2]->Set(g / sqrt(2) / a2, dt_1malpha_modes_gf[4]);  
      break;
    }
    default:
      MFEM_ABORT("Only dimensions 2 and 3 are allowed.")
      break;
  }
}

void TimeFractionalFokkerPlanckSolver::UpdateTimestepHistory(double dt) {
  // Evaluate fractional next fractional mode 
  if(alpha == 1.0){
    // nothing to do here 
  } else {
    if(non_zero_initial){
      // modes are directly updated as part of the SDIRK2 scheme 
    }else{
      for (int k = 0; k < m; ++k) {
        modes_k_next[k].Set(gamma_k[k],modes_k[k]); 
        modes_k_next[k].Add(delta_k[k],modes_k_prev[k]); 
        modes_k_next[k].Add(beta1_k[k],modes_next);     
      }
    }

    // Update the current fractional modes 
    for (int k = 0; k < m; ++k) {
      modes_k_prev[k].Set(1.,modes_k[k]); 
      modes_k[k].Set(1.,modes_k_next[k]); 
    }

    // Update the modes and corresponding GridFunctions 
    for (int k = 0; k < m; ++k) {
      modes_k[k] = modes_k_next[k];
      for (auto &gf : modes_k_gf[k]){
        gf.SetFromTrueVector();   // if(first_iteration){
  //   first_iteration = false; 
  //   for (int i = 0; i < settings.GetNModes(dim) * (settings.GetNModes(dim) - 1); ++i) {
  //     Cmat_old[i] = Add(0., *Cmat[i], 1., *Cmat[i]); 
  //   }
  // }

      } 
    }
  }

  // Update the current solution and corresponding GridFunctions
  modes_prev = modes; 
  modes = modes_next;

  for (auto &gf : modes_gf) {
    gf.SetFromTrueVector();
  }

  // Update the extra stress grid functions 
  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      *extra_stress_gf[i][j] = *extra_stress_next_gf[i][j];
    }
  }
}

void TimeFractionalFokkerPlanckSolver::SetModeEquationParameters(double dt){
  sum_of_weights_m_poles_beta1 = 0.; 
  sum_of_poles_beta2 = 0.; 

  // BDF2 
  if(true){
    for (int k = 0; k < m; ++k){
      delta_k[k] = - 1 / (3 + 2 * dt * poles[k]);
      gamma_k[k] = 4 / (3 + 2 * dt * poles[k]); 
      beta1_k[k] = 2 * dt * weights[k] / (3 + 2 * dt * poles[k]);

      sum_of_weights_m_poles_beta1 += weights[k] - poles[k] * beta1_k[k]; 
    }
  }
}

void TimeFractionalFokkerPlanckSolver::ExtrapolateExtraStress(){
  // This is only applicable for constant time steps dt 
  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      extra_stress_np2_gf[i][j]->Set(2.,  *extra_stress_next_gf[i][j]);   
      extra_stress_np2_gf[i][j]->Add(-1., *extra_stress_gf[i][j]);   
    }
  }
}

void TimeFractionalFokkerPlanckSolver::SetWeightsPoles(){

  if(alpha == 0.2){
    weights[0] = 0.6391212563714556;
    weights[1] = 0.24756350983358633;
    weights[2] = 0.19780552697975162;
    weights[3] = 0.1841704608247481;
    weights[4] = 0.18437062852149474;
    weights[5] = 0.1925588035424941;
    weights[6] = 0.2064176505049454;
    weights[7] = 0.2248082957524572;
    weights[8] = 0.2471966374708174;
    weights[9] = 0.2735114857235346;
    weights[10] = 0.30417176113207384;
    weights[11] = 0.34025312584555495;
    weights[12] = 0.3838342892882894;
    weights[13] = 0.4387099280617482;
    weights[14] = 0.5121073288678755;
    weights[15] = 0.6191771378264089;
    weights[16] = 0.7958337429296456;
    weights[17] = 1.1437184798533657;
    weights[18] = 2.0607751463347395;
    weights[19] = 6.9856765150625515;

    poles[0] = 0.017728410133439357;
    poles[1] = 0.3781993744828005;
    poles[2] = 1.3114483294155312;
    poles[3] = 3.1526910879581482;
    poles[4] = 6.504499963208466;
    poles[5] = 12.421610855649938;
    poles[6] = 22.74161921595204;
    poles[7] = 40.65226826229362;
    poles[8] = 71.66830851602806;
    poles[9] = 125.33525516954904;
    poles[10] = 218.26400497966333;
    poles[11] = 379.72431436963296;
    poles[12] = 662.4838655646267;
    poles[13] = 1165.2793027163427;
    poles[14] = 2083.923418316693;
    poles[15] = 3843.791502488275;
    poles[16] = 7512.120323786966;
    poles[17] = 16466.177998275285;
    poles[18] = 46811.419288112964;
    poles[19] = 307164.87903683906;

    w_inf = 0.; 
  }

  if(alpha == 0.5){ // m = 20
    weights[0] = 0.29495162548497;
    weights[1] = 0.32832875570781894;
    weights[2] = 0.39019867628107335;
    weights[3] = 0.4815596040349227;
    weights[4] = 0.6099111457409042;
    weights[5] = 0.7866079846215535;
    weights[6] = 1.0258843063172205;
    weights[7] = 1.3450948399606728;
    weights[8] = 1.7658958016844049;
    weights[9] = 2.3170666249577208;
    weights[10] = 3.040151916302446;
    weights[11] = 3.9993790316988345;
    weights[12] = 5.301008160976108;
    weights[13] = 7.1363030354606405;
    weights[14] = 9.885322428734066;
    weights[15] = 14.401535192450554;
    weights[16] = 22.943383761055706;
    weights[17] = 43.15846016799679;
    weights[18] = 115.69416335031995;
    weights[19] = 1019.8942580349675;

    poles[0] = 0.05259950286814615;
    poles[1] = 0.5110359834414007;
    poles[2] = 1.6271492897453648;
    poles[3] = 3.8260932392852904;
    poles[4] = 7.883556144321019;
    poles[5] = 15.189712521184077;
    poles[6] = 28.213972522037487;
    poles[7] = 51.29553797065258;
    poles[8] = 91.9776353146568;
    poles[9] = 163.27375827501854;
    poles[10] = 287.62164908985073;
    poles[11] = 504.0803896999915;
    poles[12] = 882.1931268430998;
    poles[13] = 1550.9242847990517;
    poles[14] = 2765.9452900359383;
    poles[15] = 5089.721304897402;
    poles[16] = 9974.596194981068;
    poles[17] = 22257.531156166107;
    poles[18] = 67476.62109165263;
    poles[19] = 634619.2664935145;

    w_inf = 0;  
  }

  if(alpha == 0.8){
    weights[0] = 0.088657690209173;
    weights[1] = 0.18155195378477895;
    weights[2] = 0.30399856887125126;
    weights[3] = 0.4851506124988676;
    weights[4] = 0.7612217206341917;
    weights[5] = 1.1853764744637942;
    weights[6] = 1.8367710291151498;
    weights[7] = 2.8342897527259034;
    weights[8] = 4.359307110214214;
    weights[9] = 6.695220952546737;
    weights[10] = 10.298409979941297;
    weights[11] = 15.932353152772095;
    weights[12] = 24.944635845232902;
    weights[13] = 39.90856195816055;
    weights[14] = 66.31399653602121;
    weights[15] = 117.76711105817995;
    weights[16] = 235.76861429191646;
    weights[17] = 594.6543964575097;
    weights[18] = 2553.8004752182114;
    weights[19] = 140796.23027635732;
    
    poles[0] = 0.09929195922555772;
    poles[1] = 0.6757466940630236;
    poles[2] = 2.017337860758069;
    poles[3] = 4.640300158155587;
    poles[4] = 9.461972994112125;
    poles[5] = 18.079732586789042;
    poles[6] = 33.24350036166239;
    poles[7] = 59.64717463519374;
    poles[8] = 105.27031382496861;
    poles[9] = 183.70503602667182;
    poles[10] = 318.3172867342598;
    poles[11] = 549.9779528587133;
    poles[12] = 952.2093167251309;
    poles[13] = 1663.2751865996202;
    poles[14] = 2962.130344991053;
    poles[15] = 5477.020322121377;
    poles[16] = 10887.171249162668;
    poles[17] = 25120.340359016147;
    poles[18] = 83762.03684840623;
    poles[19] = 1744557.4282361786;
    
    w_inf = 0;  
  }
}

void TimeFractionalFokkerPlanckSolver::SetInitialConditions2D(Coefficient &phi00, 
                                                              Coefficient &phi02, 
                                                              Coefficient &phi11, 
                                                              // if(first_iteration){
  //   first_iteration = false; 
  //   for (int i = 0; i < settings.GetNModes(dim) * (settings.GetNModes(dim) - 1); ++i) {
  //     Cmat_old[i] = Add(0., *Cmat[i], 1., *Cmat[i]); 
  //   }
  // }
  Coefficient &phi20){
  modes_gf[0].ProjectCoefficient(phi00); 
  modes_gf[1].ProjectCoefficient(phi02); 
  modes_gf[2].ProjectCoefficient(phi11); 
  modes_gf[3].ProjectCoefficient(phi20); 

  modes_gf[0].SetTrueVector(); 
  modes_gf[1].SetTrueVector(); 
  modes_gf[2].SetTrueVector(); 
  modes_gf[3].SetTrueVector(); 

  non_zero_initial = true; 
}

TimeFractionalFokkerPlanckSolver::~TimeFractionalFokkerPlanckSolver() {
  for (int i = 0; i < pmesh.Dimension(); ++i) {
    for (int j = 0; j < pmesh.Dimension(); ++j) {
      delete extra_stress_gf[i][j];
      delete extra_stress_next_gf[i][j];
      delete extra_stress_np2_gf[i][j]; 
    }
  }
  delete fp_operator;
  delete ode_solver;
}