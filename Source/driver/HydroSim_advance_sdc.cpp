
#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <Diffusion.H>
#endif

#include <cmath>
#include <climits>

using std::string;
using namespace amrex;

#ifndef MHD
#ifndef AMREX_USE_GPU
Real
HydroSim::run_advance_sdc (Real time,
                        Real dt,
                        int  amr_pass,
                        int  amr_cycles)
{


    amrex::ignore_unused(amr_pass);
    amrex::ignore_unused(amr_cycles);


    BL_PROFILE("HydroSim::run_advance_sdc()");

    const Real prev_time = state[State_Type].prevTime();
    const Real  cur_time = state[State_Type].curTime();

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& S_new = get_new_data(State_Type);

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    step_result status {};


    status = begin_step_advance(time, dt);

    MultiFab& old_forcing = get_old_data(Forcing_Type);
    MultiFab& new_source = get_new_data(Forcing_Type);

    bool apply_source_flag = false;


    for (int m=0; m < COLLOC_NODES; m++) {

        current_colloc_node = m;


        Real sdc_node_time = time + node_dt_frac[m]*dt;

        MultiFab::Copy(S_new, *(node_new[m]), 0, 0, S_new.nComp(), 0);
        clean_state(S_new, cur_time, 0);
        fill_ghost_field(Sborder, cur_time, NUM_GROW);


        if (!(correction_iter > 0 && m == 0) &&
            !(correction_iter == sdc_accuracy_order+sdc_extra_passes-1 && m == COLLOC_NODES-1)) {


#ifdef GRAVITY
            assemble_old_gravity(prev_time);
#endif

            if (apply_source_terms()) {
                if (sdc_accuracy_order == 4) {

                    for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {
                        const Box& gbx = mfi.growntilebox(1);

                        build_center_state(gbx, Sborder.array(mfi), BurnData.array(mfi), domain_lo, domain_hi);

                    }

                    run_old_sources(old_forcing, BurnData, BurnData, sdc_node_time, dt, apply_source_flag);

                    AmrLevel::FillPatch(*this, old_forcing, old_forcing.nGrow(), prev_time, Forcing_Type, 0, NUM_SOURCE);

                    FArrayBox tmp;

                    for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {
                        const Box& bx = mfi.tilebox();

                        tmp.resize(bx, 1, The_Async_Arena());
                        auto tmp_arr = tmp.array();

                        build_fourth_in_place(bx, old_forcing.array(mfi), tmp_arr, domain_lo, domain_hi);
                    }

                } else {
                    run_old_sources(old_forcing, Sborder, Sborder, cur_time, dt, apply_source_flag);
                }

                if (sdc_accuracy_order == 2 && use_pressure_slope == 1) {
                    AmrLevel::FillPatch(*this, old_forcing, old_forcing.nGrow(), prev_time, Forcing_Type, 0, NUM_SOURCE);
                }
            }



            if (sdc_accuracy_order == 4) {
                make_primitives_fourth(time);
            } else {
                make_primitives(time);
            }

            if (do_hydro) {
                verify_courant_limit(S_old, dt);
            }

            Advect_new[m]->setVal(0.0);
            assemble_mol_hydro_source(time, dt, *Advect_new[m]);

        }

        if (correction_iter == 0 && m == 0) {
            for (int n=1; n < COLLOC_NODES; n++) {
                MultiFab::Copy(*(Advect_old[n]), *(Advect_new[0]), 0, 0, NUM_STATE, 0);
            }

#ifdef REACTIONS

            bool input_is_mean = true;
            assemble_old_react_source(Sborder, *(Reactions_old[0]), input_is_mean);

            for (int n = 1; n < COLLOC_NODES; n++) {
                MultiFab::Copy(*(Reactions_old[n]), *(Reactions_old[0]), 0, 0, Reactions_old[0]->nComp(), 0);
            }
#endif
        }

        if (m < COLLOC_NODES-1) {

            amrex::Print() << "... doing the SDC update, iteration = " << correction_iter << " from node " << m << " to " << m+1 << std::endl;

            do_sdc_correct(m, m+1, dt);

            clean_state(*(node_new[m+1]), cur_time, 0);

        }


    }



    if (correction_iter != sdc_accuracy_order+sdc_extra_passes-1) {
        for (int n=1; n < COLLOC_NODES; n++) {
            MultiFab::Copy(*(Advect_old[n]), *(Advect_new[n]), 0, 0, NUM_STATE, 0);
        }
    }

#ifdef REACTIONS

    for (int m = 1; m < COLLOC_NODES; ++m) {
        MultiFab::Copy(S_new, *(node_new[m]), 0, 0, S_new.nComp(), 0);
        fill_ghost_field(BurnData, cur_time, 2);
        bool input_is_mean = true;
        assemble_old_react_source(BurnData, *(Reactions_old[m]), input_is_mean);
    }
#endif

    if (correction_iter == sdc_accuracy_order+sdc_extra_passes-1) {

        MultiFab::Copy(S_new, *(node_new[COLLOC_NODES-1]), 0, 0, S_new.nComp(), 0);



        clean_state(S_old, prev_time, 0);
        fill_ghost_field(Sborder, prev_time, Sborder.nGrow());
        run_old_sources(old_forcing, Sborder, Sborder, prev_time, dt, apply_source_flag);
        AmrLevel::FillPatch(*this, old_forcing, old_forcing.nGrow(), prev_time, Forcing_Type, 0, NUM_SOURCE);

        clean_state(S_new, cur_time, 0);
        fill_ghost_field(Sborder, cur_time, Sborder.nGrow());
        run_old_sources(new_source, Sborder, Sborder, cur_time, dt, apply_source_flag);
        AmrLevel::FillPatch(*this, new_source, new_source.nGrow(), cur_time, Forcing_Type, 0, NUM_SOURCE);
    }

    status = end_step_advance(cur_time, dt);

#ifdef REACTIONS

    MultiFab& react_new = get_new_data(Burning_Type);

    if (sdc_accuracy_order == 4) {
        Sborder.define(grids, dmap, NUM_STATE, NUM_GROW, MFInfo().SetTag("Sborder"));

        fill_ghost_field(Sborder, cur_time, 2);
    }

    FArrayBox state_center;
    FArrayBox react_center;
    FArrayBox tmp;

    for (MFIter mfi(react_new); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Box& obx = mfi.growntilebox(1);

        if (sdc_accuracy_order == 4) {



            react_center.resize(obx, react_new.nComp(), The_Async_Arena());
            auto const react_center_arr = react_center.array();

            Array4<const Real> const burn_state_arr = BurnData.array(mfi);

            hs_store_reaction_state(obx, burn_state_arr, react_center_arr);

            tmp.resize(bx, 1, The_Async_Arena());
            auto const tmp_arr = tmp.array();

            build_fourth_in_place(bx, react_center_arr, tmp_arr, domain_lo, domain_hi);

            react_new[mfi].copy(react_center, bx, 0, bx, 0, react_new.nComp());

        } else {

            Array4<const Real> const react_old_arr = Reactions_old[COLLOC_NODES-1]->array(mfi);
            Array4<Real> const react_new_arr = react_new.array(mfi);

            hs_store_reaction_state(bx, react_old_arr, react_new_arr);
        }

    }

    if (sdc_accuracy_order == 4) {
        Sborder.clear();
    }

#endif

    return dt;
}

#endif
#endif
