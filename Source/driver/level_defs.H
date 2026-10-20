#ifndef LEVEL_DEFS_H
#define LEVEL_DEFS_H

#include <string>
#include <AMReX_REAL.H>

enum stepper_kind : std::uint8_t {
    CornerTrace = 0,
    ReservedIntegrator,
    IterativeCorrection,
    SimplifiedIterCorrection
};

enum class state_epoch : std::uint8_t { old_time, new_time };

enum class reflux_origin : std::uint8_t { post_timestep, subcycle_advance };

struct step_result {
    bool success{true};
    amrex::Real proposed_dt{};
    std::string reason{};
    step_result() = default;
};

enum StateType : std::uint8_t {
               State_Type = 0,
               Forcing_Type,
};

enum sources : std::uint8_t {
               thermodynamic_src = 0,
#ifdef SPONGE
               damping_src,
#endif
               external_source,
               geometry_source,
               num_forcing };

#endif
