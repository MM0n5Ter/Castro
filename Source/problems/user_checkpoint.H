#ifndef user_checkpoint_H
#define user_checkpoint_H

///
/// Perform problem-specific checkpoint functionality
///
/// @param checkpoint_location   Name of the checkpoint directory
///
AMREX_INLINE
void user_checkpoint (const std::string& checkpoint_location) {

    amrex::ignore_unused(checkpoint_location);

}

#endif
