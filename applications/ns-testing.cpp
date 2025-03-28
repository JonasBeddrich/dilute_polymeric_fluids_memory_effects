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

void vel_ic_case7(const Vector &x, double t, Vector &u){
    u(0) = 0.01 * sin(17 * M_1_PI * x[0]) * sin(19 * M_1_PI * x[1]) * sin(21 * M_1_PI * x[2]); 
    u(1) = 0.01 * cos(71 * M_1_PI * x[0]) * sin(37 * M_1_PI * x[1]) * sin(13 * M_1_PI * x[2]); 
    u(2) = 0.01 * sin(11 * M_1_PI * x[0]) * cos(11 * M_1_PI * x[1]) * cos(31 * M_1_PI * x[2]); 
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

    std::string pathname("NS_testing_");
    pathname += "_dt_" + std::to_string(setup.time_settings.dt);
    pathname += "_kv_" + std::to_string(setup.ns_kin_viscosity);
    pathname += "_g_" + std::to_string(setup.fp_settings.stress_iso_coeff);
    ParaViewDataCollection dataCollection(pathname, &pmesh);
    dataCollection.SetPrefixPath("ParaView/NS_Case" + std::to_string(setup.run_case) + "_");
    dataCollection.SetLevelsOfDetail(setup.fp_settings.order);
    dataCollection.SetDataFormat(VTKFormat::BINARY);
    dataCollection.SetHighOrderOutput(true);

    NavierSolver2 flowsolver(&pmesh, setup);
    ParGridFunction &velocity_gf = *flowsolver.GetCurrentVelocity();
    ParGridFunction &pressure_gf = *flowsolver.GetCurrentPressure();

    VectorFunctionCoefficient u_ic5_coef(pmesh.Dimension(), vel_ic_case5); 
    VectorFunctionCoefficient u_ic7_coef(pmesh.Dimension(), vel_ic_case7); 

    if(setup.run_case==static_cast<int>(NSFP_RunSetup::RunCase::RW_CHANNEL)){
        velocity_gf.ProjectCoefficient(u_ic5_coef); 
    }
    if(setup.run_case==static_cast<int>(NSFP_RunSetup::RunCase::PSR_3D)){
        velocity_gf.ProjectCoefficient(u_ic7_coef); 
    }

    dataCollection.RegisterField("u", &velocity_gf);
    dataCollection.RegisterField("p", &pressure_gf);

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
        if (Mpi::Root())
            {
                out << "step " << std::setw(6) << step
                    << ",  t = " << std::fixed << std::setprecision(4) << t;
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

        if (last_step || (step % output_steps) == 0)
        {
            dataCollection.SetCycle(step);
            dataCollection.SetTime(t);
            dataCollection.Save();
        }
    }

    sw_all.Stop();
    double my_rt = sw_all.RealTime();
    double max_rt;
    MPI_Reduce(&my_rt, &max_rt, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh.GetComm());

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