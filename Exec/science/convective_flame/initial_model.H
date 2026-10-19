#ifndef INITIAL_MODEL_H
#define INITIAL_MODEL_H



namespace cf {
    constexpr Real MAX_ITER = 250;
    constexpr Real TOL = 1.e-10_rt;
}

struct model_t {

    Real xn_base[NumSpec];
    Real xn_star[NumSpec];
    Real xn_perturb[NumSpec];

    Real density_base;
    Real T_star;
    Real T_hi;
    Real T_lo;

    Real H_star;
    Real atm_delta;

    Real low_density_cutoff;
};


AMREX_INLINE
void
generate_initial_model(const int npts_model, const Real xmin, const Real xmax,
                       const model_t model_params, const int model_num)
{




    model::npts = npts_model;
    model::initialized = true;

    if (npts_model > NPTS_MODEL) {
        amrex::Error("Error: model has more than NPTS_MODEL points,  Increase MAX_NPTS_MODEL");
    }


    Real dx = (xmax - xmin) / npts_model;

    for (int i = 0; i < npts_model; i++) {
        model::profile(model_num).r(i) =
            xmin + (static_cast<Real>(i) + 0.5_rt) * dx;
    }



    int index_base = -1;
    for (int i = 0; i < npts_model; i++) {
        if (model::profile(model_num).r(i) >= xmin + model_params.H_star) {
            index_base = i+1;
            break;
        }
    }

    if (index_base == -1) {
        amrex::Error("ERROR: invalid base_height");
    }



    bool fluff = false;


    eos_t eos_state;
    eos_state.T = model_params.T_star;
    eos_state.rho = model_params.density_base;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = model_params.xn_star[n];
    }

    eos(eos_input_rt, eos_state);


    Real pres_base = eos_state.p;


    for (int i = 0; i < npts_model; i++) {

        Real xc = model::profile(model_num).r(i) -
            (xmin + model_params.H_star) - 1.5_rt * model_params.atm_delta;


        for (int n = 0; n < NumSpec; n++) {
            model::profile(model_num).state(i, model::ispec+n) =
                model_params.xn_star[n] +
                0.5_rt * (model_params.xn_base[n] - model_params.xn_star[n]) *
                (1.0_rt + std::tanh(xc / (0.5_rt * model_params.atm_delta)));
        }


        Real sumX = 0.0_rt;
        for (int n = 0; n < NumSpec; n++) {
            sumX += model::profile(model_num).state(i, model::ispec+n);
        }
        for (int n = 0; n < NumSpec; n++) {
            model::profile(model_num).state(i, model::ispec+n) /= sumX;
        }


        if (i <= index_base) {
            model::profile(model_num).state(i, model::itemp) = model_params.T_star;
        } else {
            model::profile(model_num).state(i, model::itemp) =
                model_params.T_star +
                0.5_rt * (model_params.T_hi - model_params.T_star) *
                (1.0_rt + std::tanh(xc / (0.5_rt * model_params.atm_delta)));
        }


        model::profile(model_num).state(i, model::idens) = model_params.density_base;
        model::profile(model_num).state(i, model::ipres) = pres_base;

    }


    eos_state.rho = model::profile(model_num).state(index_base, model::idens);
    eos_state.T = model::profile(model_num).state(index_base, model::itemp);
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = model::profile(model_num).state(index_base, model::ispec+n);
    }

    eos(eos_input_rt, eos_state);

    model::profile(model_num).state(index_base, model::ipres) = eos_state.p;




    bool isentropic = false;
    bool flipped = false;


    Real entropy_base;

    for (int i = index_base+1; i < npts_model; i++) {

       if ((model::profile(model_num).r(i) > xmin + model_params.H_star + 3.0_rt * model_params.atm_delta) && !flipped) {
           isentropic = true;
           flipped = true;


           eos_state.rho = model::profile(model_num).state(i-1, model::idens);
           eos_state.T = model::profile(model_num).state(i-1, model::itemp);
           for (int n = 0; n < NumSpec; n++) {
               eos_state.xn[n] = model::profile(model_num).state(i-1, model::ispec+n);
           }

           eos(eos_input_rt, eos_state);

           entropy_base = eos_state.s;

           amrex::Print() << "base density = " << eos_state.rho << " " << eos_state.T << std::endl;
       }


       Real dens_zone = model::profile(model_num).state(i, model::idens);
       Real temp_zone = model::profile(model_num).state(i, model::itemp);
       Real xn[NumSpec];
       for (int n = 0; n < NumSpec; n++) {
           xn[n] = model::profile(model_num).state(i, model::ispec+n);
       }




       bool hydrostatic_converged = false;

       Real pressure_zone;
       Real entropy;

       if (!fluff) {

           Real p_want;
           Real drho;
           Real dtemp = 0;

           for (int iter = 0; iter < cf::MAX_ITER; iter++) {




               p_want = model::profile(model_num).state(i-1, model::ipres) +
                   dx * 0.5_rt * (dens_zone + model::profile(model_num).state(i-1, model::idens)) * gravity::uniform_accel;

               if (isentropic) {




                   eos_state.T = temp_zone;
                   eos_state.rho = dens_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       eos_state.xn[n] = xn[n];
                   }

                   eos(eos_input_rt, eos_state);

                   entropy = eos_state.s;
                   pressure_zone = eos_state.p;

                   Real dpt = eos_state.dpdT;
                   Real dpd = eos_state.dpdr;
                   Real dst = eos_state.dsdT;
                   Real dsd = eos_state.dsdr;

                   Real A = p_want - pressure_zone;
                   Real B = entropy_base - entropy;

                   dtemp = ((dsd / (dpd - 0.5_rt * dx * gravity::uniform_accel)) * A - B) /
                       (dsd * dpt / (dpd - 0.5_rt * dx * gravity::uniform_accel) - dst);

                   drho = (A - dpt * dtemp) / (dpd - 0.5_rt * dx * gravity::uniform_accel);

                   dens_zone = amrex::max(0.9_rt * dens_zone,
                                          amrex::min(dens_zone + drho, 1.1_rt * dens_zone));

                   temp_zone = amrex::max(0.9_rt * temp_zone,
                                          amrex::min(temp_zone + dtemp, 1.1_rt * temp_zone));


                   if (dens_zone < model_params.low_density_cutoff) {

                       dens_zone = model_params.low_density_cutoff;
                       temp_zone = model_params.T_lo;
                       hydrostatic_converged = true;
                       fluff = true;
                       break;
                   }

                   if (std::abs(drho) < cf::TOL * dens_zone && std::abs(dtemp) < cf::TOL * temp_zone) {
                       hydrostatic_converged = true;
                       break;
                   }

               } else {


                   if (model::profile(model_num).r(i) > xmin + model_params.H_star + 3.0_rt * model_params.atm_delta) {
                       temp_zone = model_params.T_lo;
                   }


                   eos_state.T = temp_zone;
                   eos_state.rho = dens_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       eos_state.xn[n] = xn[n];
                   }

                   eos(eos_input_rt, eos_state);

                   entropy = eos_state.s;
                   pressure_zone = eos_state.p;

                   Real dpd = eos_state.dpdr;

                   drho = (p_want - pressure_zone) / (dpd - 0.5_rt * dx * gravity::uniform_accel);

                   dens_zone = amrex::max(0.9_rt * dens_zone,
                                          amrex::min(dens_zone + drho, 1.1_rt * dens_zone));

                   if (std::abs(drho) < cf::TOL * dens_zone) {
                       hydrostatic_converged = true;
                       break;
                   }

                   if (dens_zone < model_params.low_density_cutoff) {
                       dens_zone = model_params.low_density_cutoff;
                       temp_zone = model_params.T_lo;
                       hydrostatic_converged = true;
                       fluff = true;
                       break;
                   }

               }

               if (temp_zone < model_params.T_lo) {
                   temp_zone = model_params.T_lo;
                   isentropic = false;
               }

           }

           if (!hydrostatic_converged) {
               std::cout << "Error zone " << i << " did not converge in init_1d" << std::endl;
               std::cout << "integrate up" << std::endl;
               std::cout << dens_zone << " " << temp_zone << std::endl;
               std::cout << p_want << " " << entropy_base << " " << entropy << std::endl;
               std::cout << drho << " " << dtemp << std::endl;
               amrex::Error("Error: HSE non-convergence");
          }

       } else {

           dens_zone = model_params.low_density_cutoff;
           temp_zone = model_params.T_lo;
       }



       eos_state.T = temp_zone;
       eos_state.rho = dens_zone;
       for (int n = 0; n < NumSpec; n++) {
           eos_state.xn[n] = xn[n];
       }

       eos(eos_input_rt, eos_state);

       pressure_zone = eos_state.p;


       model::profile(model_num).state(i, model::idens) = dens_zone;
       model::profile(model_num).state(i, model::itemp) = temp_zone;
       model::profile(model_num).state(i, model::ipres) = pressure_zone;


    }



    for (int i = index_base-1; i >= 0; --i) {


        Real temp_zone = model::profile(model_num).state(i, model::itemp);
        Real xn[NumSpec];
        for (int n = 0; n < NumSpec; n++) {
            xn[n] = model::profile(model_num).state(i, model::ispec+n);
        }


        Real dens_zone = model::profile(model_num).state(i+1, model::idens);



        bool hydrostatic_converged = false;

        Real pressure_zone;
        Real p_want;
        Real drho;

        for (int iter = 0; iter < cf::MAX_ITER; iter++) {



            p_want = model::profile(model_num).state(i+1, model::ipres) -
                dx * 0.5_rt * (dens_zone + model::profile(model_num).state(i+1, model::idens)) * gravity::uniform_accel;


            eos_state.T = temp_zone;
            eos_state.rho = dens_zone;
            for (int n = 0; n < NumSpec; n++) {
                eos_state.xn[n] = xn[n];
            }

            eos(eos_input_rt, eos_state);

            pressure_zone = eos_state.p;

            Real dpd = eos_state.dpdr;

            Real A = p_want - pressure_zone;

            drho = A / (dpd + 0.5_rt * dx * gravity::uniform_accel);

            dens_zone = amrex::max(0.9_rt * dens_zone,
                                   amrex::min(dens_zone + drho, 1.1_rt * dens_zone));

            if (std::abs(drho) < cf::TOL * dens_zone) {
                hydrostatic_converged = true;
                break;
            }

        }

        if (!hydrostatic_converged) {
            std::cout << "Error zone " << i << " did not converge in init_1d" << std::endl;
            std::cout << "integrate down" << std::endl;
            std::cout << dens_zone << " " << temp_zone << std::endl;
            std::cout << p_want << std::endl;
            std::cout << drho << std::endl;
            amrex::Error("Error: HSE non-convergence");
        }



        eos_state.T = temp_zone;
        eos_state.rho = dens_zone;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = xn[n];
        }

        eos(eos_input_rt, eos_state);

        pressure_zone = eos_state.p;


        model::profile(model_num).state(i, model::idens) = dens_zone;
        model::profile(model_num).state(i, model::itemp) = temp_zone;
        model::profile(model_num).state(i, model::ipres) = pressure_zone;

    }
}
#endif
