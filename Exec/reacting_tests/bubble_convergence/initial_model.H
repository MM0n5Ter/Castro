
#ifndef INITIAL_MODEL_H
#define INITIAL_MODEL_H

#include <network.H>
#include <eos.H>

struct model_t {
    Real density_base = -1;
    Real T_base = -1;
    Real xn[NumSpec] = {0.0};
};

AMREX_INLINE
Real
f(const Real p, const Real s, const Real g, const Real rho, const Real T, const Real* xn) {

    eos_t eos_state;

    eos_state.T = T;
    eos_state.rho = rho;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.p = p;
    eos_state.s = s;

    eos(eos_input_ps, eos_state);

    return eos_state.rho * g;
}


AMREX_INLINE
void
generate_initial_model(const int nx, const Real xmin, const Real xmax,
                       const model_t model_params, const int nbuf) {

    constexpr int MAX_ITER = 100;
    constexpr Real TOL = 1.e-10_rt;


    int npts_model = nx + 2*nbuf;

    model::npts = npts_model;
    model::initialized = true;

    int ibase = nbuf;
    int itop = ibase + nx - 1;


    eos_t eos_state;
    eos_state.rho = model_params.density_base;
    eos_state.T = model_params.T_base;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = model_params.xn[n];
    }

    eos(eos_input_rt, eos_state);

    Real entropy_fixed = eos_state.s;


    Real dx = (xmax - xmin) / nx;

    for (int i = 0; i < npts_model; i++) {
        model::profile(0).r(i) = xmin + (static_cast<Real>(i - nbuf) + 0.5_rt) * dx;
    }


    Real p = eos_state.p;

    if (time_scheme == 2 && sdc_accuracy_order == 4) {



        for (int i = ibase; i <= itop + nbuf; i++) {


            Real rho;
            Real T;

            if (i == ibase) {
                rho = eos_state.rho;
                T = eos_state.T;
            } else {
                rho = model::profile(0).state(i-1, model::idens);
                T = model::profile(0).state(i-1, model::itemp);
            }

            Real xn[NumSpec];
            for (int n = 0; n < NumSpec; n++) {
                xn[n] = model_params.xn[n];
            }


            Real h;
            if (i == ibase) {
                h = 0.5_rt * dx;
            } else {
                h = dx;
            }


            Real s = entropy_fixed;

            Real k1 = f(p, s, gravity::uniform_accel, rho, T, xn);
            Real k2 = f(p + 0.5_rt * h * k1, s, gravity::uniform_accel, rho, T, xn);
            Real k3 = f(p + 0.5_rt * h * k2, s, gravity::uniform_accel, rho, T, xn);
            Real k4 = f(p + h * k3, s, gravity::uniform_accel, rho, T, xn);

            Real pnew = p + (1.0_rt/6.0_rt) * h * (k1 + 2.0_rt * k2 + 2.0_rt * k3 + k4);


            eos_state.T = T;
            eos_state.rho = rho;
            for (int n = 0; n < NumSpec; n++) {
                eos_state.xn[n] = model_params.xn[n];
            }
            eos_state.p = pnew;
            eos_state.s = s;

            eos(eos_input_ps, eos_state);


            model::profile(0).state(i, model::idens) = eos_state.rho;
            model::profile(0).state(i, model::itemp) = eos_state.T;
            model::profile(0).state(i, model::ipres) = eos_state.p;
            for (int n = 0; n < NumSpec; n++) {
                model::profile(0).state(i, model::ispec+n) = eos_state.xn[n];
            }


            p = pnew;

        }


        p = model::profile(0).state(ibase, model::ipres);

        for (int i = ibase-1; i >= 0; --i) {

            Real rho = model::profile(0).state(i+1, model::idens);
            Real T = model::profile(0).state(i+1, model::itemp);

            Real h = -dx;


            Real s = entropy_fixed;

            Real k1 = f(p, s, gravity::uniform_accel, rho, T, model_params.xn);
            Real k2 = f(p + 0.5_rt * h * k1, s, gravity::uniform_accel, rho, T, model_params.xn);
            Real k3 = f(p + 0.5_rt * h * k2, s, gravity::uniform_accel, rho, T, model_params.xn);
            Real k4 = f(p + h * k3, s, gravity::uniform_accel, rho, T, model_params.xn);

            Real pnew = p + (1.0_rt/6.0_rt) * h * (k1 + 2.0_rt*k2 + 2.0_rt*k3 + k4);

            eos_state.T = T;
            eos_state.rho = rho;
            for (int n = 0; n < NumSpec; n++) {
                eos_state.xn[n] = model_params.xn[n];
            }
            eos_state.p = pnew;
            eos_state.s = s;

            eos(eos_input_ps, eos_state);


            model::profile(0).state(i, model::idens) = eos_state.rho;
            model::profile(0).state(i, model::itemp) = eos_state.T;
            model::profile(0).state(i, model::ipres) = eos_state.p;
            for (int n = 0; n < NumSpec; n++) {
                model::profile(0).state(i, model::ispec+n) = eos_state.xn[n];
            }


            p = pnew;

        }

    } else {


        for (int i = ibase; i <= itop + nbuf; i++) {

            if (i == ibase) {
                model::profile(0).state(i, model::idens) = model_params.density_base;
                model::profile(0).state(i, model::ipres) =
                    p + 0.5_rt * dx * model_params.density_base * gravity::uniform_accel;

                eos_state.rho = model::profile(0).state(i, model::idens);
                eos_state.T = model_params.density_base;
                eos_state.p = model::profile(0).state(i, model::ipres);
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = model_params.xn[n];
                }
                eos_state.s = entropy_fixed;

                eos(eos_input_ps, eos_state);

                model::profile(0).state(i, model::idens) = eos_state.rho;
                model::profile(0).state(i, model::itemp) = eos_state.T;
                for (int n = 0; n < NumSpec; n++) {
                    model::profile(0).state(i, model::ispec+n) = model_params.xn[n];
                }

            } else {

                Real dens_zone = model::profile(0).state(i-1, model::idens);

                bool hydrostatic_converged = false;

                Real p_want;

                for (int iter = 0; iter < MAX_ITER; iter++) {

                    p_want = model::profile(0).state(i-1, model::ipres) +
                        dx * 0.5_rt * (dens_zone + model::profile(0).state(i-1, model::idens)) * gravity::uniform_accel;


                    eos_state.p = p_want;
                    eos_state.s = entropy_fixed;
                    eos_state.rho = dens_zone;
                    eos_state.T = model_params.T_base;
                    for (int n = 0; n < NumSpec; n++) {
                        eos_state.xn[n] = model_params.xn[n];
                    }

                    eos(eos_input_ps, eos_state);

                    Real drho = eos_state.rho - dens_zone;
                    dens_zone = eos_state.rho;
                    if (std::abs(drho) < TOL * dens_zone) {
                        hydrostatic_converged = true;
                        break;
                    }

                }

                if (!hydrostatic_converged) {
                    amrex::Error("ERROR: failed to convergence in initial model generation");
                }


                model::profile(0).state(i, model::idens) = dens_zone;
                model::profile(0).state(i, model::ipres) = p_want;
                model::profile(0).state(i, model::itemp) = eos_state.T;
                for (int n = 0; n < NumSpec; n++) {
                    model::profile(0).state(i, model::ispec+n) = model_params.xn[n];
                }


            }

            amrex::Print() << i << " " << model::profile(0).state(i, model::idens) << " " << model::profile(0).state(i, model::ipres) << std::endl;
        }


        for (int i = ibase-1; i >= 0; --i) {

            Real dens_zone = model::profile(0).state(i+1, model::idens);

            bool hydrostatic_converged = false;

            Real p_want;

            for (int iter = 0; iter < MAX_ITER; iter++) {

                p_want = model::profile(0).state(i+1, model::ipres) -
                    dx * 0.5_rt * (dens_zone + model::profile(0).state(i+1, model::idens)) * gravity::uniform_accel;


                eos_state.p = p_want;
                eos_state.s = entropy_fixed;
                eos_state.rho = dens_zone;
                eos_state.T = model_params.T_base;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = model_params.xn[n];
                }

                eos(eos_input_ps, eos_state);

                Real drho = eos_state.rho - dens_zone;
                dens_zone = eos_state.rho;
                if (std::abs(drho) < TOL * dens_zone) {
                    hydrostatic_converged = true;
                    break;
                }
            }

            if (!hydrostatic_converged) {
                amrex::Error("failed to convergence in initial model generation (down)");
            }


            model::profile(0).state(i, model::idens) = dens_zone;
            model::profile(0).state(i, model::ipres) = p_want;
            model::profile(0).state(i, model::itemp) = eos_state.T;
            for (int n = 0; n < NumSpec; n++) {
                model::profile(0).state(i, model::ispec+n) = model_params.xn[n];
            }

        }
    }
}

#endif

