#ifndef PROBLEM_H
#define PROBLEM_H

/* problem-specific HydroSim:: declarations go here */

#ifdef DO_PROBLEM_POST_TIMESTEP
void user_post_timestep();
#endif

#ifdef DO_PROBLEM_POST_RESTART
void user_post_restart();
#endif

#ifdef DO_PROBLEM_POST_INIT
void user_post_init();
#endif

#endif
