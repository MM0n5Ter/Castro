#ifndef USER_H
#define USER_H

/* problem-specific HydroSim:: declarations go here */

#ifdef DO_USER_POST_TIMESTEP
void user_post_timestep();
#endif

#ifdef DO_USER_POST_RESTART
void user_post_restart();
#endif

#ifdef DO_USER_POST_INIT
void user_post_init();
#endif

#endif
