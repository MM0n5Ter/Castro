
#include <new>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <iomanip>

#ifndef WIN32
#include <unistd.h>
#endif

#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_AmrLevel.H>

#include <ctime>

#include <HydroSim.H>
#include <HydroSim_io.H>

#include <global.H>

using namespace amrex;

std::string inputs_name{};

amrex::LevelBld* getLevelBld ();

// Any parameters we want to override the defaults for in AMReX

void set_amrex_defaults ()
{
    {
        ParmParse pp("amrex");
#ifndef RADIATION // Radiation is not yet ready to stop using managed memory
        if (!pp.contains("the_arena_is_managed")) {
            // Use device memory allocations, not managed memory.
            pp.add("the_arena_is_managed", false);
        }
#endif
        if (!pp.contains("abort_on_out_of_gpu_memory")) {
            // Abort if we run out of GPU memory.
            pp.add("abort_on_out_of_gpu_memory", true);
        }
    }

    {
        ParmParse pp("amr");
        // Always check for whether to dump a plotfile or checkpoint.
        if (!pp.contains("message_int")) {
            pp.add("message_int", 1);
        }
    }
}

int
main (int   argc,
      char* argv[])
{

    // check to see if it contains --describe
    if (argc >= 2) {
        for (auto i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--describe") {
                HydroSim::writeBuildRecord();
                return 0;
            }
        }
    }

    //
    // Make sure to catch new failures.
    //
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, set_amrex_defaults);
    {

    // Refuse to continue if we did not provide an inputs file.

    if (argc <= 1) {
        amrex::Abort("Error: no inputs file provided on command line.");
    }

    // Save the inputs file name for later.

    if (!strchr(argv[1], '=')) {
        inputs_name = argv[1];
    }

    BL_PROFILE_VAR("main()", pmain);

    double dRunTime1 = ParallelDescriptor::second();

    std::cout << std::setprecision(10);

    int  max_step;
    Real strt_time;
    Real stop_time;
    ParmParse pp;

    max_step  = -1;
    strt_time =  0.0;
    stop_time = -1.0;

    pp.query("max_step",max_step);
    pp.query("strt_time",strt_time);
    pp.query("stop_time",stop_time);

    if (strt_time < 0.0)
    {
        amrex::Abort("MUST SPECIFY a non-negative strt_time");
    }

    if (max_step < 0 && stop_time < 0.0) {
      amrex::Abort(
       "Exiting because neither max_step nor stop_time is non-negative.");
    }

    // Print the current date and time.

    time_t time_type;

    const struct tm* wall_tm_ptr = nullptr;

    time(&time_type);

    wall_tm_ptr = gmtime(&time_type);

    amrex::Print() << std::setfill('0') << "\nStarting run at "
                   << std::setw(2) << wall_tm_ptr->tm_hour << ":"
                   << std::setw(2) << wall_tm_ptr->tm_min << ":"
                   << std::setw(2) << wall_tm_ptr->tm_sec << " UTC on "
                   << wall_tm_ptr->tm_year + 1900 << "-"
                   << std::setw(2) << wall_tm_ptr->tm_mon + 1 << "-"
                   << std::setw(2) << wall_tm_ptr->tm_mday << "." << std::endl;

    //
    // Initialize random seed after we're running in parallel.
    //

    Amr* amr_obj = new Amr(getLevelBld());
    global::amr_instance = amr_obj;

    amr_obj->init(strt_time,stop_time);

    // If we set the regrid_on_restart flag and if we are *not* going to take
    //    a time step then we want to go ahead and regrid here.
    if ( amr_obj->RegridOnRestart() &&
         ( (amr_obj->levelSteps(0) >= max_step) ||
           (amr_obj->cumTime() >= stop_time) ) )
           {
           //
           // Regrid only!
           //
           amr_obj->RegridOnly(amr_obj->cumTime());
           }

    double dRunTime2 = ParallelDescriptor::second();

    while ( amr_obj->okToContinue()                            &&
           (amr_obj->levelSteps(0) < max_step || max_step < 0) &&
           (amr_obj->cumTime() < stop_time || stop_time < 0.0) )

    {
        //
        // Do a timestep.
        //
        amr_obj->coarseTimeStep(stop_time);
    }

#ifdef DO_PROBLEM_POST_SIMULATION
    HydroSim::user_post_simulation(amr_obj->getAmrLevels());
#endif

    // Write final checkpoint and plotfile

    if (HydroSim::get_write_on_finish() == 1) {

        if (amr_obj->stepOfLastCheckPoint() < amr_obj->levelSteps(0)) {
            amr_obj->checkPoint();
        }

        if (amr_obj->stepOfLastPlotFile() < amr_obj->levelSteps(0)) {
            amr_obj->writePlotFile();
        }

        if (amr_obj->stepOfLastSmallPlotFile() < amr_obj->levelSteps(0)) {
            amr_obj->writeSmallPlotFile();
        }

    }

    // Start calculating the figure of merit for this run: average number of zones
    // advanced per microsecond. This must be done before we delete the Amr
    // object because we need to scale it by the number of zones on the coarse grid.

    long coarse_zone_count = amr_obj->getLevel(0).boxArray().numPts();
    Real fom = HydroSim::cells_advanced_total * static_cast<Real>(coarse_zone_count);

    time(&time_type);

    wall_tm_ptr = gmtime(&time_type);

    amrex::Print() << std::setfill('0') << "\nEnding run at "
                   << std::setw(2) << wall_tm_ptr->tm_hour << ":"
                   << std::setw(2) << wall_tm_ptr->tm_min << ":"
                   << std::setw(2) << wall_tm_ptr->tm_sec << " UTC on "
                   << wall_tm_ptr->tm_year + 1900 << "-"
                   << std::setw(2) << wall_tm_ptr->tm_mon + 1 << "-"
                   << std::setw(2) << wall_tm_ptr->tm_mday << "." << std::endl;

    delete amr_obj;
    //
    // This MUST follow the above delete as ~Amr() may dump files to disk.
    //
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    const int nprocs = ParallelDescriptor::NProcs();

    double end_wall_time = ParallelDescriptor::second();

    Real wall_time_total = static_cast<Real>(end_wall_time - dRunTime1);
    Real wall_time_solve = static_cast<Real>(end_wall_time - dRunTime2);

    ParallelDescriptor::ReduceRealMax(wall_time_total,IOProc);
    ParallelDescriptor::ReduceRealMax(wall_time_solve,IOProc);

    if (ParallelDescriptor::IOProcessor())
    {
        std::cout << "Run time = " << wall_time_total << std::endl;
        std::cout << "Run time without initialization = " << wall_time_solve << std::endl;

        fom = fom / wall_time_solve / 1.e6;

        std::cout << "\n";
        std::cout << "  Average number of zones advanced per microsecond: " << std::fixed << std::setprecision(3) << fom << "\n";
        std::cout << "  Average number of zones advanced per microsecond per rank: " << std::fixed << std::setprecision(3) << fom / nprocs << "\n";
        std::cout << "\n";
    }

    if (auto* arena = dynamic_cast<CArena*>(amrex::The_Arena()))
    {
        //
        // A barrier to make sure our output follows that of RunStats.
        //
        ParallelDescriptor::Barrier();
        //
        // We're using a CArena -- output some FAB memory stats.
        //
        // This'll output total # of bytes of heap space in the Arena.
        //
        // It's actually the high water mark of heap space required by FABs.
        //
        std::cout << std::format("CPU({}): Heap Space (bytes) used by Coalescing FAB Arena: {}\n",
                                 ParallelDescriptor::MyProc(), arena->heap_space_used());
    }

    BL_PROFILE_VAR_STOP(pmain);
    BL_PROFILE_SET_RUN_TIME(dRunTime2);

    }
    amrex::Finalize();

    return 0;
}
