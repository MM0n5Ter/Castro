

#ifndef DO_USER_POST_INIT
#define DO_USER_POST_INIT
#endif

#ifndef DO_USER_POST_RESTART
#define DO_USER_POST_RESTART
#endif

#ifndef DO_USER_POST_TIMESTEP
#define DO_USER_POST_TIMESTEP
#endif


void user_post_init();


void user_post_timestep();


void user_post_restart();


amrex::Real compute_moment_of_inertia();


static amrex::Real initial_moment_of_inertia;
