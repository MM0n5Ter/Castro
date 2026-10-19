#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <HydroSim.H>

#include <particles_params.H>

#include <particles_globals.H>

using namespace amrex;

#ifdef AMREX_PARTICLES

std::unique_ptr<AmrTracerParticleContainer> HydroSim::TracerPC;

namespace {
    std::vector<int>  timestamp_component_ids;
    const std::string tracer_checkpoint_label("Tracer");
}

void
HydroSim::load_particle_config ()
{

  ParmParse pp("particles");

    if (ParallelDescriptor::IOProcessor())
        if (!amrex::UtilCreateDirectory(particles::snapshot_dir, 0755))
            amrex::CreateDirectoryFailed(particles::snapshot_dir);
    ParallelDescriptor::Barrier();
}

void
HydroSim::init_particles ()
{
    BL_PROFILE("HydroSim::init_particles()");

    if (level > 0)
        return;

    if (enable_tracer_particles)
    {
        BL_ASSERT(TracerPC == nullptr);

        TracerPC = std::make_unique<AmrTracerParticleContainer>(parent);

        TracerPC->SetVerbose(particles::particle_log_level);

        if (! particles::particle_seed_file.empty())
        {
            TracerPC->InitFromAsciiFile(particles::particle_seed_file,0);
        }
    }
}

void
HydroSim::ParticleSaveRestart(const std::string& dir)
{
    if (level == 0)
    {
        if (TracerPC)
            TracerPC->Checkpoint(dir, tracer_checkpoint_label);
    }
}

void
HydroSim::ParticleOutputFile(const std::string& dir)
{
    if (level == 0)
    {
        if (TracerPC)
            TracerPC->Checkpoint(dir, tracer_checkpoint_label);
    }
}

void
HydroSim::ParticleAfterRestart (const std::string& restart_path)
{
    if (level == 0)
    {
        if (enable_tracer_particles)
        {
            BL_ASSERT(TracerPC == nullptr);

            TracerPC = std::make_unique<AmrTracerParticleContainer>(parent);

            TracerPC->SetVerbose(particles::particle_log_level);
            if (!particles::restart_chk_no_particles)
            {
                TracerPC->Restart(parent->theRestartFile(), tracer_checkpoint_label);
            }

            if (!particles::particle_resume_file.empty())
            {
                TracerPC->InitFromAsciiFile(particles::particle_resume_file,0);
            }

            if (!particles::particle_dump_file.empty())
            {
                TracerPC->WriteAsciiFile(particles::particle_dump_file);
            }
        }
    }
}

std::unique_ptr<MultiFab>
HydroSim::ParticleQuantity(const std::string& name,
                       Real               time,
                       int                ngrow)
{
    BL_PROFILE("HydroSim::ParticleQuantity()");

  if (TracerPC && name == "particle_count")
  {
      auto derived_mf = new MultiFab(grids,dmap,1,0);
      MultiFab    fine_temp_mf(grids,dmap,1,0);
      fine_temp_mf.setVal(0);
      TracerPC->Increment(fine_temp_mf,level);
      MultiFab::Copy(*derived_mf,fine_temp_mf,0,0,1,0);
      return std::unique_ptr<MultiFab>(derived_mf);
  }
  else if (TracerPC && name == "total_particle_count")
  {
      auto derived_mf = ParticleQuantity("particle_count",time,ngrow);

      IntVect trr(AMREX_D_DECL(1,1,1));

      for (int lev = level+1; lev <= parent->finestLevel(); lev++)
      {
          BoxArray ba = parent->boxArray(lev);
          const DistributionMapping& dm = parent->DistributionMap(lev);

          MultiFab fine_temp_mf(ba,dm,1,0);

          trr *= parent->refRatio(lev-1);

          ba.coarsen(trr);

          MultiFab coarse_temp_mf(ba,dm,1,0);

          fine_temp_mf.setVal(0);
          coarse_temp_mf.setVal(0);

          TracerPC->Increment(fine_temp_mf,lev);

          for (MFIter mfi(fine_temp_mf); mfi.isValid(); ++mfi)
          {
              const FArrayBox& ffab =  fine_temp_mf[mfi];
              FArrayBox&       cfab = coarse_temp_mf[mfi];
              const Box&       fbx  = ffab.box();

              BL_ASSERT(cfab.box() == amrex::coarsen(fbx,trr));

              for (IntVect p = fbx.smallEnd(); p <= fbx.bigEnd(); fbx.next(p))
              {
                  const Real val = ffab(p);
                  if (val > 0)
                      cfab(amrex::coarsen(p,trr)) += val;
              }
          }

          fine_temp_mf.clear();

          MultiFab dat(grids,dmap,1,0);
          dat.setVal(0);
          dat.copy(coarse_temp_mf);

          MultiFab::Add(*derived_mf,dat,0,0,1,0);
      }

      return derived_mf;
  }
  else
  {
     return AmrLevel::derive(name,time,ngrow);
  }
}

void
HydroSim::StampParticles (int ngrow)
{
    static bool first = true;
    static int imax = -1;
    if (first)
    {
        first = false;

        if (particles::snapshot_density) {
            timestamp_component_ids.push_back(URHO);
            std::cout << "Density = " << URHO << std::endl;
        }
        if (particles::snapshot_temperature) {
            timestamp_component_ids.push_back(UTEMP);
            std::cout << "Temp = " << UTEMP << std::endl;
        }

        if (!timestamp_component_ids.empty()) {
            imax = *(std::max_element(timestamp_component_ids.begin(), timestamp_component_ids.end()));
        }
    }

    if ( TracerPC && !particles::snapshot_dir.empty())
    {
        std::string basename = particles::snapshot_dir;

        if (basename[basename.length()-1] != '/') basename += '/';

        basename += "Timestamp";

        int finest_level = parent->finestLevel();
        Real time        = state[State_Type].curTime();

        for (int lev = level; lev <= finest_level; lev++)
        {
            if (TracerPC->NumberOfParticlesAtLevel(lev) <= 0) continue;

            MultiFab& S_new = parent->getLevel(lev).get_new_data(State_Type);

            if (imax >= 0) {
                int ng = (lev == level) ? ngrow : 1;
                FillPatchIterator fpi(parent->getLevel(lev), S_new,
                                      ng, time, State_Type, 0, imax+1);
                const MultiFab& S = fpi.get_mf();
                TracerPC->Timestamp(basename, S    , lev, time, timestamp_component_ids);
            } else {
                TracerPC->Timestamp(basename, S_new, lev, time, timestamp_component_ids);
            }
        }
    }
}

#endif

void
HydroSim::step_particles(int iteration, Real time, Real dt)
{
    if (TracerPC)
    {
        int ng = iteration;
        Real t = time + 0.5*dt;

        MultiFab Ucc(grids,dmap,AMREX_SPACEDIM,ng);

        {
            FillPatchIterator fpi(*this, Ucc, ng, t, State_Type, 0, AMREX_SPACEDIM+1);
            MultiFab& S = fpi.get_mf();

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi(Ucc,true); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.growntilebox();
                S[mfi].invert(1.0, bx, 0, 1);
                for (int dir=0; dir < AMREX_SPACEDIM; ++dir) {
                    Ucc[mfi].copy(S[mfi], bx, dir+1, bx, dir, 1);
                    Ucc[mfi].mult(S[mfi], bx, 0, dir);
                }
            }
        }

        TracerPC->AdvectWithUcc(Ucc, level, dt);
    }
}
