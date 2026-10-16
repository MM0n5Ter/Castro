#ifndef user_restart_H
#define user_restart_H

///
/// Perform problem-specific restart functionality
///
/// @param checkpoint_dir   Name of the checkpoint directory
///
AMREX_INLINE
void user_restart (const std::string& checkpoint_dir) {

    amrex::ignore_unused(checkpoint_dir);

}

#endif
