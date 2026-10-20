

#ifndef DO_USER_POST_INIT
#define DO_USER_POST_INIT
#endif


#ifndef DO_USER_POST_RESTART
#define DO_USER_POST_RESTART
#endif


#ifndef DO_USER_POST_TIMESTEP
#define DO_USER_POST_TIMESTEP
#endif


void wd_update(amrex::Real time, amrex::Real dt);


struct StellarVolumes {
    amrex::Real primary;
    amrex::Real secondary;
};

StellarVolumes volInBoundary (amrex::Real time, amrex::Real rho_cutoff, bool local = false);


static
amrex::Real dot_product(const amrex::Real a[], const amrex::Real b[]);



static
amrex::Real norm(const amrex::Real a[]);


void user_post_init();


void user_post_restart();


void user_post_timestep();


static
void writeGitHashes(std::ostream& log);


void update_relaxation(amrex::Real time, amrex::Real dt);


void problem_sums();
