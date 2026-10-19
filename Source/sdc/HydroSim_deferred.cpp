#include <HydroSim.H>
#include <HydroSim_deferred_util.H>

using namespace amrex;

void
HydroSim::do_sdc_correct(int m_start, int m_end, Real dt)
{

    BL_PROFILE("HydroSim::do_sdc_correct()");

    // NOTE: dt here is the full dt not the dt between time nodes

    // this routine needs to do the update from time node m to m+1
    //
    // We come in with:
    //   Advect_new[m_start] : this is the advective update at node m_start
    //   Advect_old[:] : this is the advective source for all nodes at the old iterate
    //
    //   Reactions_old[:] : this is the reaction source for all nodes at the old iterate

    // If we do advection only, then the update is explicit.  If we do
    // reactions, then the update is implicit within a zone.

    // for 4th order reactive SDC, we need to first compute the source, C
    // and do a ghost cell fill on it

#ifdef REACTIONS
    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();
#endif

    // the timestep from m to m+1
    Real substep_dt = (node_dt_frac[m_end] - node_dt_frac[m_start]) * dt;

#ifdef REACTIONS
    // Forcing_Type_Index is only defined for 4th order
    MultiFab tmp;
    MultiFab& Correction_source = (sdc_accuracy_order == 4) ? get_new_data(Forcing_Type_Index) : tmp;

    if (sdc_accuracy_order == 4) {

        // for 4th order reacting flow, we need to create the "source" C
        // as averages and then convert it to cell centers.  The cell-center
        // version needs to have 2 ghost cells
        for (MFIter mfi(*node_new[0]); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();
            Array4<Real> const& Correction_source_arr = Correction_source.array(mfi);

            Array4<const Real> const& Advect_new_arr = (Advect_new[m_start])->array(mfi);
            Array4<const Real> const& Advect_old_0_arr = (Advect_old[0])->array(mfi);
            Array4<const Real> const& Advect_old_1_arr = (Advect_old[1])->array(mfi);
            Array4<const Real> const& Advect_old_2_arr = (Advect_old[2])->array(mfi);
            Array4<const Real> const& React_old_0_arr = (Reactions_old[0])->array(mfi);
            Array4<const Real> const& React_old_1_arr = (Reactions_old[1])->array(mfi);
            Array4<const Real> const& React_old_2_arr = (Reactions_old[2])->array(mfi);

            if (sdc_quadrature == 0) {
                hs_deferred_compute_correction_o4_lobatto(bx, substep_dt, dt, Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr,
                                          Advect_old_2_arr, React_old_0_arr, React_old_1_arr, React_old_2_arr,
                                          Correction_source_arr, m_start);

            } else {
                Array4<const Real> const& Advect_old_3_arr = (Advect_old[3])->array(mfi);
                Array4<const Real> const& React_old_3_arr = (Reactions_old[3])->array(mfi);

                hs_deferred_compute_correction_o4_radau(bx, substep_dt, dt, Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr,
                                        Advect_old_2_arr,
                                        Advect_old_3_arr, React_old_0_arr, React_old_1_arr, React_old_2_arr, React_old_3_arr,
                                        Correction_source_arr, m_start);

            }
        }

        // need to construct the time for this stage -- but it is not really
        // at a single instance in time.  For single level this does not matter,
        Real time = state[Forcing_Type_Index].curTime();
        AmrLevel::FillPatch(*this, Correction_source, Correction_source.nGrow(), time,
                            Forcing_Type_Index, 0, NUM_STATE);


        // we'll also construct an initial guess for the nonlinear solve,
        // and store this in the BurnData MultiFab.  We'll use S_new as the
        // staging place so we can do a FillPatch
        MultiFab& S_new = get_new_data(State_Type);

        for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            Array4<const Real> const& node_state_new_start_arr = (node_new[m_start])->array(mfi);
            Array4<const Real> const& node_state_new_end_arr = (node_new[m_end])->array(mfi);
            Array4<const Real> const& Advect_old_arr = (Advect_old[m_start])->array(mfi);
            Array4<const Real> const& react_old_arr = (Reactions_old[m_start])->array(mfi);
            Array4<Real> const& new_state_view = S_new.array(mfi);

            hs_deferred_compute_initial_guess(bx, node_state_new_start_arr, node_state_new_end_arr,
                                         Advect_old_arr, react_old_arr, new_state_view,
                                         substep_dt, correction_iter);

        }

        const Real cur_time = state[State_Type].curTime();
        fill_ghost_field(BurnData, cur_time, 2);

    }
#endif

    // main update loop -- we are updating node_new[m_start] to
    // node_new[m_end]

    FArrayBox state_center(The_Async_Arena());
    FArrayBox Correction_center(The_Async_Arena());
    FArrayBox State_new_center(The_Async_Arena());
    FArrayBox react_new(The_Async_Arena());
    FArrayBox laplacian_work(The_Async_Arena());

    FArrayBox C2(The_Async_Arena());

    for (MFIter mfi(*node_new[0]); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

#ifdef REACTIONS
        const Box& bx1 = mfi.growntilebox(1);

        // advection + reactions
        if (sdc_accuracy_order == 2) {

            // second order SDC reaction update -- we don't care about
            // the difference between cell-centers and averages

            // first compute the source term, C -- this differs depending
            // on whether we are Lobatto or Radau
            C2.resize(bx, NUM_STATE);
            Array4<Real> const& Correction_o2_arr = C2.array();

            Array4<const Real> const& Advect_new_arr = (Advect_new[m_start])->array(mfi);
            Array4<const Real> const& Advect_old_0_arr = (Advect_old[0])->array(mfi);
            Array4<const Real> const& Advect_old_1_arr = (Advect_old[1])->array(mfi);
            Array4<const Real> const& React_old_0_arr = (Reactions_old[0])->array(mfi);
            Array4<const Real> const& React_old_1_arr = (Reactions_old[1])->array(mfi);

            if (sdc_quadrature == 0) {
                hs_deferred_compute_correction_o2_lobatto(bx, substep_dt, dt, Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr,
                                          React_old_0_arr, React_old_1_arr, Correction_o2_arr, m_start);
            } else {
                Array4<const Real> const& Advect_old_2_arr = (Advect_old[2])->array(mfi);
                Array4<const Real> const& React_old_2_arr = (Reactions_old[2])->array(mfi);
                hs_deferred_compute_correction_o2_radau(bx, substep_dt, dt, Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr,
                                        Advect_old_2_arr,
                                        React_old_0_arr, React_old_1_arr, React_old_2_arr, Correction_o2_arr, m_start);
            }

            auto k_m = (*node_new[m_start]).array(mfi);
            auto k_n = (*node_new[m_end]).array(mfi);
            auto A_m = (*Advect_new[m_start]).array(mfi);
            auto React_m_old = (*Reactions_old[m_start]).array(mfi);
            auto Correction_arr = C2.array();

            auto sdc_pass_number = correction_iter;   // avoid capturing this
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // solve U^{n+1} - substep_dt R(U^{n+1}) - U^n - substep_dt C = 0
                // we use A_m and React_m_old to construct an initial guess for U^{n+1}
                deferred_update_o2(i, j, k, k_m, k_n, A_m, React_m_old, Correction_arr, substep_dt, sdc_pass_number, m_start);
            });

        } else {

            // fourth order SDC reaction update -- we need to respect the
            // difference between cell-centers and averages

            Array4<const Real> const& node_state_new_start_arr = (node_new[m_start])->array(mfi);
            Array4<Real> const& node_state_new_end_arr = (node_new[m_end])->array(mfi);
            Array4<const Real> const& Correction_source_arr = Correction_source.array(mfi);

            // convert the starting U to cell-centered on a fab-by-fab basis
            // -- including one ghost cell
            state_center.resize(bx1, NUM_STATE);
            auto State_center_arr = state_center.array();

            build_center_state(bx1, Sborder.array(mfi), State_center_arr, domain_lo, domain_hi);

            // sometimes the Laplacian can make the species go negative near discontinuities
            amrex::ParallelFor(bx1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                normalize_species_deferred(i, j, k, State_center_arr);
            });

            // convert the C source to cell-centers
            Correction_center.resize(bx1, NUM_STATE);
            auto Correction_center_arr = Correction_center.array();

            build_center_state(bx1, Correction_source.array(mfi), Correction_center_arr, domain_lo, domain_hi);

            // solve for the updated cell-center U using our cell-centered C -- we
            // need to do this with one ghost cell
            State_new_center.resize(bx1, NUM_STATE);
            auto State_new_center_arr = State_new_center.array();

            // initialize cons_new with our guess for the new state, stored as
            // an average in BurnData
            build_center_state(bx1, BurnData.array(mfi), State_new_center_arr, domain_lo, domain_hi);

            auto sdc_pass_number = correction_iter;  // avoid capturing this

            amrex::ParallelFor(bx1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                deferred_update_centers_o4(i, j, k, State_center_arr, State_new_center_arr, Correction_center_arr, substep_dt, sdc_pass_number);
            });

            // enforce that the species sum to one after the reaction solve
            amrex::ParallelFor(bx1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                normalize_species_deferred(i, j, k, State_new_center_arr);
            });

            // compute R_i and in 1 ghost cell and then convert to <R> in
            // place (only for the interior)
            react_new.resize(bx1, NUM_STATE);
            Array4<Real> const& react_new_arr = react_new.array();

            amrex::ParallelFor(bx1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                instant_reaction_rate(i, j, k, State_new_center_arr, react_new_arr);
            });

            laplacian_work.resize(bx, 1);
            auto const laplacian_work_arr = laplacian_work.array();

            build_fourth_in_place(bx, react_new_arr, laplacian_work_arr, domain_lo, domain_hi);

            // now do the conservative update using this <R> to get <U>
            // We'll also need to pass in <C>
            hs_deferred_conservative_update(bx, substep_dt, node_state_new_start_arr, node_state_new_end_arr,
                                       Correction_source_arr, react_new_arr);

        }
#else
        Array4<const Real> const& node_state_new_start_arr = (node_new[m_start])->array(mfi);
        Array4<Real> const& node_state_new_end_arr = (node_new[m_end])->array(mfi);
        Array4<const Real> const& Advect_new_arr = (Advect_new[m_start])->array(mfi);
        Array4<const Real> const& Advect_old_0_arr = (Advect_old[0])->array(mfi);
        Array4<const Real> const& Advect_old_1_arr = (Advect_old[1])->array(mfi);

        // pure advection
        if (sdc_accuracy_order == 2) {

            if (sdc_quadrature == 0) {
                hs_deferred_update_advection_o2_lobatto(bx, substep_dt, dt, node_state_new_start_arr,
                                                   node_state_new_end_arr,
                                                   Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr,
                                                   m_start);

            } else {
                Array4<const Real> const& Advect_old_2_arr = (Advect_old[2])->array(mfi);
                hs_deferred_update_advection_o2_radau(bx, substep_dt, dt, node_state_new_start_arr,
                                                 node_state_new_end_arr,
                                                 Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr, Advect_old_2_arr,
                                                 m_start);

            }

        } else {
            Array4<const Real> const& Advect_old_2_arr = (Advect_old[2])->array(mfi);
            if (sdc_quadrature == 0) {
                hs_deferred_update_advection_o4_lobatto(bx, substep_dt, dt, node_state_new_start_arr,
                                                   node_state_new_end_arr,
                                                   Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr, Advect_old_2_arr,
                                                   m_start);

            } else {
                Array4<const Real> const& Advect_old_3_arr = (Advect_old[3])->array(mfi);
                hs_deferred_update_advection_o4_radau(bx, substep_dt, dt, node_state_new_start_arr,
                                                 node_state_new_end_arr,
                                                 Advect_new_arr, Advect_old_0_arr, Advect_old_1_arr, Advect_old_2_arr,
                                                 Advect_old_3_arr, m_start);

            }

        }
#endif

    }
}


#ifdef REACTIONS
void
HydroSim::assemble_old_react_source(MultiFab& cell_cons,
                                   MultiFab& React_source,
                                   const bool input_is_mean)
{

    // note: there is a side effect here.  If we are fourth-order and
    // coming in with input_is_mean = true, then we make a backup
    // of React_source when it is on cell-centers and store it in BurnData.

    BL_PROFILE("HydroSim::assemble_old_react_source()");

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    if (sdc_accuracy_order == 4 && input_is_mean)
    {
        // we have cell-averages
        // Note: we cannot tile these operations

        FArrayBox state_center(The_Async_Arena());
        FArrayBox react_center(The_Async_Arena());
        FArrayBox tmp(The_Async_Arena());

        for (MFIter mfi(cell_cons); mfi.isValid(); ++mfi)
        {

            const Box& bx = mfi.tilebox();
            const Box& obx = mfi.growntilebox(1);

            // Convert to centers
            state_center.resize(obx, NUM_STATE);
            auto const State_center_arr = state_center.array();

            build_center_state(obx, cell_cons.array(mfi), State_center_arr, domain_lo, domain_hi);

            // sometimes the Laplacian can make the species go
            // negative near discontinuities
            amrex::ParallelFor(obx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                normalize_species_deferred(i, j, k, State_center_arr);
            });

            // burn, including one ghost cell
            react_center.resize(obx, NUM_STATE);
            auto const react_center_arr = react_center.array();

            amrex::ParallelFor(obx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                instant_reaction_rate(i, j, k, State_center_arr, react_center_arr);
            });

            // at this point, we have the reaction term on centers,
            // including a ghost cell.  Save this into BurnData so we can
            // use it later for the plotfile filling
            BurnData[mfi].copy(react_center, obx, 0, obx, 0, NUM_STATE);

            // convert R to averages (in place)

            tmp.resize(bx, 1);
            auto const tmp_arr = tmp.array();

            build_fourth_in_place(bx, react_center_arr, tmp_arr, domain_lo, domain_hi);

            // copy this to the center
            React_source[mfi].copy(react_center, bx, 0, bx, 0, NUM_STATE);
        }

    } else {
        // we are cell-centers

        for (MFIter mfi(cell_cons); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            auto const cell_cons_arr = cell_cons.array(mfi);
            auto const React_source_arr = React_source.array(mfi);

            // construct the reactive source term
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                instant_reaction_rate(i, j, k, cell_cons_arr, React_source_arr);
            });
        }
    }
}
#endif
