#ifndef MODEL_PARSER_H
#define MODEL_PARSER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ranges>

#include <network.H>
#include <model_parser_data.H>
#include <AMReX_Print.H>
#include <hydrosim_params.H>
#include <eos.H>
#include <ambient.H>

using namespace amrex;




namespace model_string
{
    inline std::string& ltrim(std::string& s)
    {
        auto it = std::ranges::find_if(s,
                                       [] (int c) {return !std::isspace(c);});
        s.erase(s.begin(), it);
        return s;
    }

    inline std::string& rtrim(std::string& s)
    {
        auto it = std::ranges::find_if(std::ranges::reverse_view(s),
                                       [] (int c) {return !std::isspace(c);});
        s.erase(it.base(), s.end());
        return s;
    }
}


AMREX_INLINE AMREX_GPU_HOST_DEVICE
int
locate(const Real r, const int model_index) {

    int loc;

    if (r <= model::profile(model_index).r(0)) {
       loc = 0;

    } else if (r > model::profile(model_index).r(model::npts-2)) {
       loc = model::npts-2;

    } else {

        int ilo = 0;
        int ihi = model::npts-2;

        while (ilo+1 != ihi) {
            int imid = (ilo + ihi) / 2;

            if (r <= model::profile(model_index).r(imid)) {
                ihi = imid;
            } else {
                ilo = imid;
            }
        }

        loc = ilo;
    }

    return loc;
}


AMREX_INLINE AMREX_GPU_HOST_DEVICE
Real
interpolate(const Real r, const int var_index, const int model_index=0) {


    int id = locate(r, model_index);

    Real slope;
    Real interp;

    slope = (model::profile(model_index).state(id+1, var_index) -
             model::profile(model_index).state(id, var_index)) /
        (model::profile(model_index).r(id+1) - model::profile(model_index).r(id));
    interp = slope * (r - model::profile(model_index).r(id)) +
        model::profile(model_index).state(id, var_index);

    if (r >= model::profile(model_index).r(0)) {
        Real stencil_min = std::min(model::profile(model_index).state(id+1, var_index),
                               model::profile(model_index).state(id, var_index));
        Real stencil_max = std::max(model::profile(model_index).state(id+1, var_index),
                               model::profile(model_index).state(id, var_index));
        interp = amrex::Clamp(interp, stencil_min, stencil_max);
    }

    return interp;

}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real interpolate_3d (const Real* loc, const Real* dx, int var_index, int nsub = 1, int model_index = 0)
{

    Real interp = 0.0_rt;

    for (int k = 0; k < nsub; ++k) {
        Real z = loc[2] + (static_cast<Real>(k) + 0.5_rt * (1 - nsub)) * dx[2] / nsub;

        for (int j = 0; j < nsub; ++j) {
            Real y = loc[1] + (static_cast<Real>(j) + 0.5_rt * (1 - nsub)) * dx[1] / nsub;

            for (int i = 0; i < nsub; ++i) {
                Real x = loc[0] + (static_cast<Real>(i) + 0.5_rt * (1 - nsub)) * dx[0] / nsub;

                Real dist = std::sqrt(x * x + y * y + z * z);

                interp += interpolate(dist, var_index, model_index);
            }
        }
    }


    interp /= (nsub * nsub * nsub);

    return interp;
}


AMREX_INLINE
void establish_hse (Real& mass_want, Real& central_density_want, Real& radius,
                    const Real core_comp[NumSpec], Real temperature, Real dx,
                    Real envelope_mass, const Real envelope_comp[NumSpec], int model_index = 0)
{


    if (mass_want < 0.0_rt && central_density_want < 0.0_rt) {
        amrex::Error("Error: Must specify either mass or central density in the initial model generator.");
    }



    const int max_hse_iter = 250;
    int max_mass_iter;

    Real rho_c, rho_c_old, drho_c;
    Real mass, mass_old;
    Real p_want, drho;

    if (central_density_want > 0.0_rt) {

        max_mass_iter = 1;

        rho_c_old = central_density_want;
        rho_c     = central_density_want;

    }
    else {

        max_mass_iter = max_hse_iter;

        rho_c_old = -1.0_rt;
        rho_c     = 1.e7_rt;

    }


    if (temperature < hydrosim::small_temp) {
        amrex::Error("Error: WD central temperature is less than small_temp. Aborting.");
    }

    bool mass_converged = false;

    model::initial_model_t& model = model::profile(model_index);

    for (int mass_iter = 1; mass_iter <= max_mass_iter; ++mass_iter) {

        bool fluff = false;


        model.state(0, model::itemp) = temperature;
        model.state(0, model::idens) = rho_c;
        for (int n = 0; n < NumSpec; ++n) {
            model.state(0, model::ispec + n) = core_comp[n];
        }

        eos_t eos_state;
        eos_state.rho  = model.state(0, model::idens);
        eos_state.T    = model.state(0, model::itemp);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = model.state(0, model::ispec + n);
        }
#ifdef AUX_THERMO
        set_aux_comp_from_X(eos_state);
#endif

        eos(eos_input_rt, eos_state);

        model.state(0, model::ipres) = eos_state.p;

        model.r(0) = 0.5 * dx;

        int icutoff = NPTS_MODEL;


        for (int i = 1; i < NPTS_MODEL; ++i) {
            model.state(i, model::idens) = model.state(0, model::idens);
            model.state(i, model::itemp) = model.state(0, model::itemp);
            model.state(i, model::ipres) = model.state(0, model::ipres);
            for (int n = 0; n < NumSpec; ++n) {
                model.state(i, model::ispec + n) = model.state(0, model::ispec + n);
            }
            model.r(i) = model.r(i-1) + dx;
        }


        Real rl = 0.0_rt;
        Real rr = rl + dx;

        Real M_enclosed = (4.0_rt / 3.0_rt) * M_PI * (std::pow(rr, 3) - std::pow(rl, 3)) * model.state(0, model::idens);
        mass = M_enclosed;

        for (int i = 1; i < NPTS_MODEL; ++i) {

            rl += dx;
            rr += dx;


            model.state(i, model::idens) = model.state(i-1, model::idens);

            if (mass_want > 0.0_rt && M_enclosed >= mass_want - envelope_mass) {
                for (int n = 0; n < NumSpec; ++n) {
                    model.state(i, model::ispec + n) = envelope_comp[n];
                    eos_state.xn[n] = model.state(i, model::ispec + n);
                }
            }
            else {
                for (int n = 0; n < NumSpec; ++n) {
                    model.state(i, model::ispec + n) = core_comp[n];
                    eos_state.xn[n] = model.state(i, model::ispec + n);
                }
            }
#ifdef AUX_THERMO
            set_aux_comp_from_X(eos_state);
#endif

            Real g = -C::Gconst * M_enclosed / (std::pow(rl, 2));




            bool hydrostatic_converged = false;

            for (int hse_iter = 1; hse_iter <= max_hse_iter; ++hse_iter) {

                if (fluff) {
                    model.state(i, model::idens) = ambient::ambient_reference_state[URHO];
                    eos_state.rho = ambient::ambient_reference_state[URHO];
                    break;
                }



                Real rho_avg = 0.5_rt * (model.state(i, model::idens) + model.state(i-1, model::idens));
                p_want = model.state(i-1, model::ipres) + dx * rho_avg * g;

                eos(eos_input_rt, eos_state);

                drho = (p_want - eos_state.p) / (eos_state.dpdr - 0.5_rt * dx * g);

                model.state(i, model::idens) = amrex::max(0.9_rt * model.state(i, model::idens),
                                                     amrex::min(model.state(i, model::idens) + drho,
                                                                1.1_rt * model.state(i, model::idens)));
                eos_state.rho = model.state(i, model::idens);

                if (model.state(i, model::idens) < ambient::ambient_reference_state[URHO]) {
                    icutoff = i;
                    fluff = true;
                }

                if (std::abs(drho) < model::hse_tol * model.state(i, model::idens)) {
                    hydrostatic_converged = true;
                    break;
                }

            }

            if (!hydrostatic_converged && (!fluff)) {

                std::cout << "Error: zone " <<  i << " did not converge in init_hse()" << std::endl;
                std::cout << " rho = " << model.state(i, model::idens)
                          << " temp = " << model.state(i, model::itemp)
                          << " p_want = " << p_want
                          << " p = " << eos_state.p
                          << " drho = " << drho
                          << " tol * dens = " << model::hse_tol * model.state(i, model::idens)
                          << std::endl;
                amrex::Error("Error: HSE non-convergence.");

            }


            eos(eos_input_rt, eos_state);

            model.state(i, model::ipres) = eos_state.p;


            Real dM = (4.0_rt / 3.0_rt) * M_PI * model.state(i, model::idens) * dx *
                      (rr * rr + rl * rr + rl * rl);
            M_enclosed += dM;

            if (i <= icutoff) {
                mass += dM;
            }

        }

        radius = model.r(icutoff);

        if (rho_c_old < 0.0_rt) {


            rho_c_old = rho_c;
            rho_c = 0.5_rt * rho_c_old;

        }
        else {


            if (std::abs(mass - mass_want) / mass_want < model::mass_tol) {
                mass_converged = true;
                break;
            }


            drho_c = (mass_want - mass) / ((mass  - mass_old) / (rho_c - rho_c_old));

            rho_c_old = rho_c;
            rho_c = amrex::min(1.1e0_rt * rho_c_old, amrex::max((rho_c + drho_c), 0.9e0_rt * rho_c_old));

        }

        mass_old = mass;

    }

    if (!mass_converged && max_mass_iter > 1) {
        amrex::Error("ERROR: WD mass did not converge.");
    }

    central_density_want = model.state(0, model::idens);
    mass_want = mass;

    model::initialized = true;
    model::npts = NPTS_MODEL;
}

AMREX_INLINE
void
read_model_file(std::string& model_file, const int model_index=0) {

    bool found_model, found_dens, found_temp, found_pres, found_velr;
    bool found_spec[NumSpec];
#if NAUX_NET > 0
    bool found_aux[NumAux];
#endif


    std::ifstream initial_model_file;

    initial_model_file.open(model_file, std::ios::in);
    if (!initial_model_file.is_open()) {
        amrex::Error("Error opening the initial model");
    }

    amrex::Print() << Font::Bold << FGColor::Green
                   << "reading initial model" << ResetDisplay << std::endl;

    std::string line;

    std::vector<std::string> varnames_stored;
    int nvars_model_file{-1};


    getline(initial_model_file, line);

    if (line.find("npts") != std::string::npos) {


        getline(initial_model_file, line);

        if (line.find("num of variables") == std::string::npos) {
            amrex::Error("Error: missing the number of variables header");
        }

        std::string num_vars_string = line.substr(line.find('=')+1, line.length());
        nvars_model_file = std::stoi(num_vars_string);


        for (int n = 0; n < nvars_model_file; n++) {
            getline(initial_model_file, line);
            std::string var_string = line.substr(line.find('#')+1, line.length());
            varnames_stored.push_back(model_string::ltrim(model_string::rtrim(var_string)));
        }

    } else {

        if (!line.empty() && line[0] == '#') {
            line = line.substr(1);
        }

        std::istringstream iss(line);


        std::string word;
        iss >> word;

        while (iss >> word) {
            varnames_stored.push_back(word);
        }

        nvars_model_file = static_cast<int>(varnames_stored.size());

    }

    amrex::Print() << Font::Bold << FGColor::Green
                   << nvars_model_file
                   << " variables found in the initial model file"
                   << ResetDisplay << std::endl;


    amrex::Vector<Real> vars_stored;
    vars_stored.resize(nvars_model_file);

    int i{0};
    while (getline(initial_model_file, line)) {

        if (i > NPTS_MODEL) {
            amrex::Error("Error: model has more than NPTS_MODEL points,  Increase MAX_NPTS_MODEL");
        }

        std::istringstream iss(line);

        iss >> model::profile(model_index).r(i);

        for (int j = 0; j < nvars_model_file; j++) {
            iss >> vars_stored[j];
        }

        for (int j = 0; j < model::nvars; j++) {
            model::profile(model_index).state(i,j) = 0.0_rt;
        }

        found_dens = false;
        found_temp = false;
        found_pres = false;
        found_velr = false;
        for (bool & e : found_spec) {
            e = false;
        }
#if NAUX_NET > 0
        for (bool & e : found_aux) {
            e = false;
        }
#endif

        for (int j = 0; j < nvars_model_file; j++) {

            found_model = false;

            if (varnames_stored[j] == "density") {
                model::profile(model_index).state(i,model::idens) = vars_stored[j];
                found_model = true;
                found_dens = true;

            } else if (varnames_stored[j] == "temperature") {
                model::profile(model_index).state(i,model::itemp) = vars_stored[j];
                found_model = true;
                found_temp = true;

            } else if (varnames_stored[j] == "pressure") {
                model::profile(model_index).state(i,model::ipres) = vars_stored[j];
                found_model = true;
                found_pres = true;

            } else if (varnames_stored[j] == "velocity") {
                model::profile(model_index).state(i,model::ivelr) = vars_stored[j];
                found_model = true;
                found_velr = true;

            } else {
                for (int comp = 0; comp < NumSpec; comp++) {
                    if (varnames_stored[j] == spec_names_cxx[comp]) {
                        model::profile(model_index).state(i,model::ispec+comp) = vars_stored[j];
                        found_model = true;
                        found_spec[comp] = true;
                        break;
                    }
                }
#if NAUX_NET > 0
                if (!found_model) {
                    for (int comp = 0; comp < NumAux; comp++) {
                        if (varnames_stored[j] == aux_names_cxx[comp]) {
                            model::profile(model_index).state(i,model::iaux+comp) = vars_stored[j];
                            found_model = true;
                            found_aux[comp] = true;
                            break;
                        }
                    }
                }
#endif
            }


            if (!found_model && i == 0) {
                amrex::Print() << Font::Bold << FGColor::Yellow
                               << "[WARNING] variable not found: " << varnames_stored[j]
                               << ResetDisplay << std::endl;
            }


        }


        if (i == 0) {
            if (!found_dens) {
                amrex::Print() << Font::Bold << FGColor::Yellow
                               << "[WARNING] density not provided in inputs file"
                               << ResetDisplay << std::endl;
            }

            if (!found_temp) {
                amrex::Print() << Font::Bold << FGColor::Yellow
                               << "[WARNING] temperature not provided in inputs file"
                               << ResetDisplay << std::endl;
            }

            if (!found_pres) {
                amrex::Print() << Font::Bold << FGColor::Yellow
                               << "[WARNING] pressure not provided in inputs file"
                               << ResetDisplay << std::endl;
            }

            if (!found_velr) {
                amrex::Print() << Font::Bold << FGColor::Yellow
                               << "[WARNING] velocity not provided in inputs file"
                               << ResetDisplay << std::endl;
            }

            for (int comp = 0; comp < NumSpec; comp++) {
                if (!found_spec[comp]) {
                    amrex::Print() << Font::Bold << FGColor::Yellow
                                   << "[WARNING] " << spec_names_cxx[comp] << " not provided in inputs file"
                                   << ResetDisplay << std::endl;
                }
            }

#if NAUX_NET > 0
            for (int comp = 0; comp < NumAux; comp++) {
                if (!found_aux[comp]) {
                    amrex::Print() << Font::Bold << FGColor::Yellow
                                   << "[WARNING] " << aux_names_cxx[comp] << " not provided in inputs file"
                                   << ResetDisplay << std::endl;
                }
            }
#endif
        }

        i++;

    }

    model::npts = i;

    amrex::Print() << Font::Bold << FGColor::Green
                   << model::npts << " points found in the initial model"
                   << ResetDisplay << std::endl;

    initial_model_file.close();

    model::initialized = true;
}


#endif
