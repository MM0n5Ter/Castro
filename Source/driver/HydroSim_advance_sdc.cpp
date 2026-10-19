
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

    // this is the new "formal" SDC integration routine.

    amrex::ignore_unused(amr_pass);
    amrex::ignore_unused(amr_cycles);

    // this does the entire update in time for 1 SDC iteration.

    BL_PROFILE("HydroSim::run_advance_sdc()");

    const Real prev_time = state[State_Type].prevTime();
    const Real  cur_time = state[State_Type].curTime();

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& S_new = get_new_data(State_Type);

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    step_result status {};

    // Perform initialization steps.

    status = begin_step_advance(time, dt);

    MultiFab& old_forcing = get_old_data(Forcing_Type);
    MultiFab& new_source = get_new_data(Forcing_Type);

    bool apply_source_flag = false;

    // we loop over all nodes, even the last, since we need to compute
    // the advective update source at each node

    for (int m=0; m < COLLOC_NODES; m++) {

        current_colloc_node = m;

        // node_new represents carries the solution.  Coming into here, it
        // will be entirely the old state, but we update it on each time
        // node in place.

        Real sdc_node_time = time + node_dt_frac[m]*dt;

        // fill Sborder with the starting node's info -- we use S_new as
        // our staging area.  Note we need to pass new_time here to the
        // FillPatch so it only pulls from the new MF -- this will not
        // work for multilevel.
        MultiFab::Copy(S_new, *(node_new[m]), 0, 0, S_new.nComp(), 0);
        clean_state(S_new, cur_time, 0);
        fill_ghost_field(Sborder, cur_time, NUM_GROW);


        // the next chunk of code constructs the advective term for the
        // current node, m.  First we get the sources, then full hydro
        // source term from the MOL driver.  Note, for m = 0, we only have
        // to do all of this the first iteration, since that state never
        // changes
        if (!(correction_iter > 0 && m == 0) &&
            !(correction_iter == sdc_accuracy_order+sdc_extra_passes-1 && m == COLLOC_NODES-1)) {

            // Construct the "old-time" sources from Sborder.  Since we are
            // working from Sborder, this will actually evaluate the sources
            // using the current stage's starting point.

            // TODO: this is not using the density at the current stage
#ifdef GRAVITY
            assemble_old_gravity(prev_time);
#endif

            if (apply_source_terms()) {
                if (sdc_accuracy_order == 4) {
                    // if we are 4th order, convert to cell-center Sborder -> Sborder_cc
                    // we'll use BurnData for this memory buffer at the moment

                    for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {
                        const Box& gbx = mfi.growntilebox(1);

                        build_center_state(gbx, Sborder.array(mfi), BurnData.array(mfi), domain_lo, domain_hi);

                    }

                    // we pass in the stage time here
                    run_old_sources(old_forcing, BurnData, BurnData, sdc_node_time, dt, apply_source_flag);

                    // fill the ghost cells for the sources -- note since we have
                    // not defined the new_source yet, we either need to copy this
                    // into new_source for the time-interpolation in the ghost
                    // fill to make sense, or so long as we are not multilevel,
                    // just use the old time (prev_time) in the fill instead of
                    // the node time (time)
                    AmrLevel::FillPatch(*this, old_forcing, old_forcing.nGrow(), prev_time, Forcing_Type, 0, NUM_SOURCE);

                    // Now convert to cell averages.  This loop cannot be tiled.
                    FArrayBox tmp;

                    for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {
                        const Box& bx = mfi.tilebox();

                        tmp.resize(bx, 1, The_Async_Arena());
                        auto tmp_arr = tmp.array();

                        build_fourth_in_place(bx, old_forcing.array(mfi), tmp_arr, domain_lo, domain_hi);
                    }

                } else {
                    // there is a ghost cell fill hidden in diffusion, so we need
                    // to pass in the time associate with Sborder
                    run_old_sources(old_forcing, Sborder, Sborder, cur_time, dt, apply_source_flag);
                }

                // note: we don't need a FillPatch on the sources, since they
                // are only used in the valid box in the conservative flux
                // update construction.  The only exception is if we are doing
                // the well-balanced method in the reconstruction of the
                // pressure.
                if (sdc_accuracy_order == 2 && use_pressure_slope == 1) {
                    AmrLevel::FillPatch(*this, old_forcing, old_forcing.nGrow(), prev_time, Forcing_Type, 0, NUM_SOURCE);
                }
            }

            // Now compute the advective term for the current node -- this
            // will be used to advance us to the next node the new time


            // Construct the primitive variables.
            if (sdc_accuracy_order == 4) {
                make_primitives_fourth(time);
            } else {
                make_primitives(time);
            }

            if (do_hydro) {
                // Check for CFL violations.
                verify_courant_limit(S_old, dt);
            }

            // construct the update for the current stage -- this fills
            // Advect_new[m] with the righthand side for this stage.
            Advect_new[m]->setVal(0.0);
            assemble_mol_hydro_source(time, dt, *Advect_new[m]);

        } // end of the m = 0 sdc_iter > 0 check

        // also, if we are the first SDC iteration, we haven't yet stored
        // any old advective terms, so we cannot yet do the quadrature
        // over nodes.  Initialize those now.  Recall, Advect_new[0] and Advect_old[0]
        // are aliased.
        if (correction_iter == 0 && m == 0) {
            for (int n=1; n < COLLOC_NODES; n++) {
                MultiFab::Copy(*(Advect_old[n]), *(Advect_new[0]), 0, 0, NUM_STATE, 0);
            }

#ifdef REACTIONS
            // if this is the first node of a new iteration, then we need
            // to compute and store the old reactive source

            // we already have the node state with ghost cells in Sborder,
            // so we can just use that as the starting point
            bool input_is_mean = true;
            assemble_old_react_source(Sborder, *(Reactions_old[0]), input_is_mean);

            // copy to the other nodes -- since the state is the same on all
            // nodes for correction_iter == 0
            for (int n = 1; n < COLLOC_NODES; n++) {
                MultiFab::Copy(*(Reactions_old[n]), *(Reactions_old[0]), 0, 0, Reactions_old[0]->nComp(), 0);
            }
#endif
        }

        // update to the next stage -- this involves computing the
        // integral over the k-1 iteration data.  Note we don't do
        // this if we are on the final node (since there is nothing to
        // update to
        if (m < COLLOC_NODES-1) {

            amrex::Print() << "... doing the SDC update, iteration = " << correction_iter << " from node " << m << " to " << m+1 << std::endl;

            do_sdc_correct(m, m+1, dt); //(node_dt_frac[m+1] - node_dt_frac[m])*dt);

            // we now have a new value of node_new[m+1], do a clean_state on it
            clean_state(*(node_new[m+1]), cur_time, 0);

        }


    } // node iteration

    // we are done with the integration over all nodes for this iteration

    // note: as of right now, S_new is not updated with the state from
    // the final time node.  This means we can still use S_new as
    // "scratch" until we finally set it.

    if (correction_iter != sdc_accuracy_order+sdc_extra_passes-1) {
        // store Advect_old for the next SDC iteration -- don't need to do n=0,
        // since that is unchanged
        for (int n=1; n < COLLOC_NODES; n++) {
            MultiFab::Copy(*(Advect_old[n]), *(Advect_new[n]), 0, 0, NUM_STATE, 0);
        }
    }

#ifdef REACTIONS
    // we just did the update, so now recompute the "old" reactive
    // source for the next SDC iteration.  We don't need to do this for
    // m = 0, since that state never changes.

    for (int m = 1; m < COLLOC_NODES; ++m) {
        // TODO: do we need a clean state here?
        MultiFab::Copy(S_new, *(node_new[m]), 0, 0, S_new.nComp(), 0);
        fill_ghost_field(BurnData, cur_time, 2);
        bool input_is_mean = true;
        assemble_old_react_source(BurnData, *(Reactions_old[m]), input_is_mean);
    }
#endif

    if (correction_iter == sdc_accuracy_order+sdc_extra_passes-1) {

        // store the new solution
        MultiFab::Copy(S_new, *(node_new[COLLOC_NODES-1]), 0, 0, S_new.nComp(), 0);

        // We need to make source_old and source_terms_new be the source terms at
        // the old and new time.  we never actually evaluated the sources
        // using the new time state (since we just constructed it).  Note:
        // we always use run_old_sources here, since we want the actual
        // source and not a correction.

        // note: we need to have ghost cells here cause some sources (in
        // particular pdivU) need them.  Perhaps it would be easier to just
        // always require State_Type to have 1 ghost cell?

        // TODO: we also need to make these 4th order!
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
    // store the reaction information as well.  Note: this will be
    // the instantaneous reactive source from the last burn.  In the
    // future, we might want to do a quadrature over Reactions_old[]

    // this is done only for the plotfile
    MultiFab& react_new = get_new_data(Burning_Type);

    if (sdc_accuracy_order == 4) {
        // fill ghost cells on S_new -- we'll need these to convert to
        // centers
        // we'll use Sborder to expand the state, but we already cleared
        // it at the end of the andance
        Sborder.define(grids, dmap, NUM_STATE, NUM_GROW, MFInfo().SetTag("Sborder"));

        fill_ghost_field(Sborder, cur_time, 2);
    }

    FArrayBox state_center;
    FArrayBox react_center;
    FArrayBox tmp;

    // this cannot be tiled
    for (MFIter mfi(react_new); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Box& obx = mfi.growntilebox(1);

        if (sdc_accuracy_order == 4) {

            // At this point, BurnData contains the cell-center reaction
            // source on one ghost-cell, since it was stored at the
            // last assemble_old_react_source.  So we can use this to
            // derive what we need.


            // pass in the reaction source at centers (burn_state_arr), including
            // one ghost cell and derive everything that is needed including
            // 1 ghost cell
            react_center.resize(obx, react_new.nComp(), The_Async_Arena());
            auto const react_center_arr = react_center.array();

            Array4<const Real> const burn_state_arr = BurnData.array(mfi);

            hs_store_reaction_state(obx, burn_state_arr, react_center_arr);

            // convert react_new from centers to averages in place
            tmp.resize(bx, 1, The_Async_Arena());
            auto const tmp_arr = tmp.array();

            build_fourth_in_place(bx, react_center_arr, tmp_arr, domain_lo, domain_hi);

            // store
            react_new[mfi].copy(react_center, bx, 0, bx, 0, react_new.nComp());

        } else {

            Array4<const Real> const react_old_arr = Reactions_old[COLLOC_NODES-1]->array(mfi);
            Array4<Real> const react_new_arr = react_new.array(mfi);

            // we don't worry about the difference between centers and averages
            hs_store_reaction_state(bx, react_old_arr, react_new_arr);
        }

    }

    if (sdc_accuracy_order == 4) {
        Sborder.clear();
    }

#endif // REACTIONS

    return dt;
}

#endif
#endif
