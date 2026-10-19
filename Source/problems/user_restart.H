#ifndef user_restart_H
#define user_restart_H

///
/// Perform problem-specific restart functionality
///
/// @param checkpoint_location   Name of the checkpoint directory
///
AMREX_INLINE
void user_restart (const std::string& checkpoint_location) {

    amrex::ignore_unused(checkpoint_location);

}

#endif
