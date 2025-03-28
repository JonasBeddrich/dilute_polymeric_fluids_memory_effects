// This program solves the Navier-Stokes-Fokker-Planck system using an spherical
// harmonics ansatz in the configuration space leading to a lower block-triangular
// system of advection-diffusion-reaction PDEs for the Fokker-Planck equation.

#define MFEM_USE_MPI

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "Coefficient.hpp"
#include "FokkerPlanckSolver.hpp"
#include "Setups.hpp"

using namespace mfem;
using namespace fokker;
using namespace navier;

void vel_ic_case5(const Vector &coords, double t, Vector &u){
    double x = coords(0);
    double y = coords(1);

    u(0) = y * y * (0.41-y) * (0.41-y) * 2 / 0.205 / 0.205 / 0.205 / 0.205; 
    u(1) = 0; 
}

int main(int argc, char *argv[])
{
    Mpi::Init();
    Hypre::Init();

    StopWatch sw_all;
    sw_all.Clear();
    sw_all.Start();

    ////////////////////////////////////////////////////////////////////////////
    // Define and read options
    NSFP_RunSetup setup;
    int output_steps = 1;

    OptionsParser args(argc, argv);
    setup.setOptions(args);
    args.AddOption(&output_steps, "-os", "--output-steps", "Output solution every n-th timestep.");
    args.ParseCheck();

    ////////////////////////////////////////////////////////////////////////////
    // initialize the runcase
    setup.initialize();
    // setup.fp_settings.print_level_fp = setup.fp_settings.print_level_fp.All();
    // setup.fp_settings.print_level_T = setup.fp_settings.print_level_T.All();
    ParMesh &pmesh = setup.GetMesh();

    H1_FECollection fec(setup.fp_settings.order, pmesh.Dimension());
    ParFiniteElementSpace fes(&pmesh, &fec);
    ParFiniteElementSpace vfes(&pmesh, &fec, pmesh.Dimension());

    if (Mpi::Root())
    {
        out << std::endl
            << "Number of elements: " << pmesh.GetNE() << std::endl
            << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Define the boundary conditions and parameter coefficients

    Array<int> dbc_bdr(pmesh.bdr_attributes.Max());    // Dirichlet BCs
    Array<int> outflow_bc(pmesh.bdr_attributes.Max()); // Outflow BCs
    dbc_bdr = 0;
    if (pmesh.bdr_attributes.Max() > 0)
    {
        // dbc_bdr[1] = 1; no Dirichlet BC for now
    }
    outflow_bc = 0;
    if (pmesh.bdr_attributes.Max() > 1)
    {
        // outflow_bc[2] = 1; // no Outflow BC for now
    }

    Array<int> domain_attr(pmesh.attributes.Max());
    domain_attr = 1;

    ////////////////////////////////////////////////////////////////////////////
    // Prepare ParaView output

    std::string pathname("NSFP_");
    switch(setup.fp_settings.configuration){
        case(FokkerPlanckSettings::Configuration::SPHERICAL): 
            pathname += std::to_string(setup.fp_settings.compute_odd_modes); 
            pathname += "_SH_"; // spherical harmonics  
            pathname += "_K" + std::to_string(setup.fp_settings.max_mode);
        break; 
        case(FokkerPlanckSettings::Configuration::TENSORIAL): 
            pathname += "_THP"; // tensorial approach hermite polynomials 
    }
    pathname += "_dt_" + std::to_string(setup.time_settings.dt);
    pathname += "_kv_" + std::to_string(setup.ns_kin_viscosity);
    pathname += "_g_" + std::to_string(setup.fp_settings.stress_iso_coeff);
    pathname += "_De_" + std::to_string(setup.fp_settings.conf_diff_coeff); 
    pathname += "_comdiff_" + std::to_string(setup.fp_settings.space_diff_coeff); 
    ParaViewDataCollection dataCollection(pathname, &pmesh);
    dataCollection.SetPrefixPath("ParaView/NSFP_Case" + std::to_string(setup.run_case));
    dataCollection.SetLevelsOfDetail(setup.fp_settings.order);
    dataCollection.SetDataFormat(VTKFormat::BINARY);
    dataCollection.SetHighOrderOutput(true);

    ////////////////////////////////////////////////////////////////////////////
    // Define the PDE/ODE solvers and add to ParaView output
    FokkerPlanckSolver fp_solver(pmesh, setup.fp_settings);
    fp_solver.AddModesToDataCollection(dataCollection);
    Array2D<ParGridFunction *> &extra_stress_gf = fp_solver.GetCurrentExtraStress();
    DivergenceMatrixGridFunctionCoefficient *div_extra_stress = new DivergenceMatrixGridFunctionCoefficient(extra_stress_gf);
    for (int i = 0; i < pmesh.Dimension(); ++i)
    {
        for (int j = 0; j < pmesh.Dimension(); ++j)
        {
            dataCollection.RegisterField("tau" + std::to_string(i) + std::to_string(j), extra_stress_gf[i][j]);
        }
    }
    ParGridFunction div_extra_stress_gf(&vfes);
    div_extra_stress_gf.ProjectCoefficient(*div_extra_stress);
    dataCollection.RegisterField("div_tau", &div_extra_stress_gf);

    NavierSolver2 flowsolver(&pmesh, setup);
    flowsolver.AddAccelTerm(div_extra_stress, domain_attr);

    ParGridFunction &velocity_gf = *flowsolver.GetCurrentVelocity();
    ParGridFunction &pressure_gf = *flowsolver.GetCurrentPressure();

    VectorFunctionCoefficient u_ic_coef(pmesh.Dimension(), vel_ic_case5); 

    if(setup.run_case==static_cast<int>(NSFP_RunSetup::RunCase::RW_CHANNEL)){
        velocity_gf.ProjectCoefficient(u_ic_coef); 
    }
    
    dataCollection.RegisterField("u", &velocity_gf);
    dataCollection.RegisterField("p", &pressure_gf);

    // Coupling to Fokker-Planck eq
    VectorGridFunctionCoefficient u_coeff(&velocity_gf);
    GradientVectorGridFunctionCoefficient grad_u(&velocity_gf);
    MatrixArrayCoefficient dudx_coeff(pmesh.Dimension());
    std::vector<ParGridFunction> dudx_gf(grad_u.GetHeight() * grad_u.GetWidth());
    for (int i = 0; i < grad_u.GetHeight(); ++i)
    {
        for (int j = 0; j < grad_u.GetWidth(); ++j)
        {
            const int idx = i * grad_u.GetWidth() + j;
            dudx_coeff.Set(i, j, new MatrixEntryCoefficient(&grad_u, i, j));
            dudx_gf[idx].SetSpace(&fes);
            dudx_gf[idx].ProjectCoefficient(*dudx_coeff.GetCoeff(i, j));
            dataCollection.RegisterField("du" + std::to_string(i) + "dx" + std::to_string(j), &dudx_gf[idx]);
        }
    }
    fp_solver.AddSpaceAdvectionTerm(domain_attr, u_coeff);
    fp_solver.AddConfigurationAdvectionTerm(domain_attr, dudx_coeff);

    fp_solver.Setup(setup.time_settings.getTimeIntegrator());
    flowsolver.Setup(setup.time_settings.dt);

    ////////////////////////////////////////////////////////////////////////////
    // Time loop
    double t = 0.0;
    const double dt = setup.time_settings.dt;
    const double t_final = setup.time_settings.t_final;

    dataCollection.SetCycle(0);
    dataCollection.SetTime(t);
    dataCollection.Save();

    bool last_step = false;
    for (int step = 1; !last_step; ++step)
    {
        if (t + dt >= t_final - dt / 2)
        {
            last_step = true;
        }

        flowsolver.Step(t, dt, step - 1);
        t -= dt; // Due to double time-integration
        fp_solver.Step(t, dt, step);

        if (last_step || (step % output_steps) == 0)
        {
            double err = 0.0;
            if (setup.has_analytic_solution())
            {
                auto *exact = setup.GetAnalyticModes(t);
                auto &sol = fp_solver.GetCurrentModes();
                for (int i = 0; i < sol.size(); ++i)
                {
                    double l2 = sol[i].ComputeL2Error(*exact->GetCoeff(i));
                    err += l2 * l2;
                }
                err = std::sqrt(err) / sol.size();
            }

            if (Mpi::Root())
            {
                out << "step " << std::setw(6) << step
                    << ",  t = " << std::fixed << std::setprecision(4) << t;
                if (setup.has_analytic_solution())
                {
                    out << ",  L2_err = " << std::scientific << std::setprecision(3) << err;
                }
                out << std::defaultfloat << std::endl;
            }

            // update ParGridFunctions for output
            for (int i = 0; i < grad_u.GetHeight(); ++i)
            {
                for (int j = 0; j < grad_u.GetWidth(); ++j)
                {
                    const int idx = i * grad_u.GetWidth() + j;
                    dudx_gf[idx].ProjectCoefficient(*dudx_coeff.GetCoeff(i, j));
                }
            }
            div_extra_stress_gf.ProjectCoefficient(*div_extra_stress);

            dataCollection.SetCycle(step);
            dataCollection.SetTime(t);
            dataCollection.Save();
        }
    }

    sw_all.Stop();
    double my_rt = sw_all.RealTime();
    double max_rt;
    MPI_Reduce(&my_rt, &max_rt, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh.GetComm());

    fp_solver.PrintTimingData();
    flowsolver.PrintTimingData();

    if (Mpi::Root())
    {
        out << std::endl
            << std::defaultfloat
            << "The simulation took " << max_rt << " seconds." << std::endl
            << std::endl
            << "Results were written to: " << dataCollection.GetPrefixPath() << pathname << std::endl;
    }

    return EXIT_SUCCESS;
}