
#include <AMReX_ParmParse.H>
#include <AMReX_LO_BCTYPES.H>
#include <RadiativeBoundary.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

int         RadiativeBoundary::first = 1;
Vector<int>  RadiativeBoundary::bc_kind(2*AMREX_SPACEDIM);
Vector<Real> RadiativeBoundary::bcval(2*AMREX_SPACEDIM);
Real        RadiativeBoundary::time = 0.0;
int         RadiativeBoundary::correction = 0;

RadiativeBoundary::RadiativeBoundary(const BoxArray& _grids, const DistributionMapping& _dmap,
                   const Geometry& _geom) :
    BaseBoundary(_grids,_dmap,1,_geom)
{
  if (first)
    init();
  first = 0;

  const BoxArray& grids = boxes();
  const Box& domain = geom.Domain();
  //  const Real* dx = geom.CellSize();
  //  const Real* xlo = geom.ProbLo();

  // It is desirable that the type array be set up after static init.
  // This is part of the reason this step is not in BaseBoundary.
  for (OrientationIter fi; fi; ++fi) {
    Orientation face = fi();
    if (bc_kind[face] == 2) {
      bc_type_fabs[face].resize(grids.size());
      for (FabSetIter bi(bndry[face]); bi.isValid(); ++bi) {
        int igrid = bi.index();
        if (domain[face] == boxes()[igrid][face] &&
            !geom.isPeriodic(face.coordDir())) {
          const Box& face_box = bndry[face][bi].box();
          bc_type_fabs[face][igrid].reset(new BaseFab<int>(face_box));
          // We don't care about the bndry values here, only the type array.
#if 0
          FORT_RADIATIVE_BOUNDARY_FILL(BL_TO_FORTRAN(bndry[face][bi]),
                         bc_type_fabs[face][igrid]->dataPtr(),
                         AMREX_ARLIM(domain.loVect()), AMREX_ARLIM(domain.hiVect()), dx, xlo, time);
#endif
        }
      }
    }
  }
}

RadiativeBoundary::RadiativeBoundary(const BoxArray& _grids, const DistributionMapping& _dmap,
                   const Geometry& _geom, Real bv) :
    BaseBoundary(_grids,_dmap,1,_geom)
{
  if (first)
    init(bv);
  first = 0;

  setBndryValues(bv);
}

void RadiativeBoundary::init()
{
  // obsolete implementation of the Marshak boundary condition requires
  // us to scale incoming fluxes by c here, since the solver classes
  // do not themselves know about c.  This c compensates
  // for the c in the diffusion coefficient at the boundaries:

//  c = Radiation::clight;

  ParmParse pp("radiation");

  Vector<int> lo_bc_kind(AMREX_SPACEDIM, 0), hi_bc_kind(AMREX_SPACEDIM, 0);
  pp.queryarr("lo_bc_kind",lo_bc_kind,0,AMREX_SPACEDIM);
  pp.queryarr("hi_bc_kind",hi_bc_kind,0,AMREX_SPACEDIM);

  Vector<Real> lo_bc_value(AMREX_SPACEDIM, 0.0), hi_bc_value(AMREX_SPACEDIM, 0.0);
  pp.queryarr("lo_bc_value",lo_bc_value,0,AMREX_SPACEDIM);
  pp.queryarr("hi_bc_value",hi_bc_value,0,AMREX_SPACEDIM);

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
    int dir = face.coordDir();
    if (face.isLow()) {
      bc_kind[face] = lo_bc_kind[dir];
      bcval[face]  = lo_bc_value[dir];
    }
    else {
      bc_kind[face] = hi_bc_kind[dir];
      bcval[face]  = hi_bc_value[dir];
    }
  }
}

void RadiativeBoundary::init(Real bv)
{
  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
    if (face.isLow()) {
      bc_kind[face] = 0;
      bcval[face]  = bv;
    }
    else {
      bc_kind[face] = 0;
      bcval[face]  = bv;
    }
  }
}

void RadiativeBoundary::setBoundaryConditions(const BCRec& bc,
                             const Geometry& geom, IntVect& ratio)
{

//  NOTE: ALL BCLOC VALUES ARE NOW DEFINED AS A LENGTH IN PHYSICAL DIMENSIONS
//        *RELATIVE* TO THE FACE, NOT IN ABSOLUTE PHYSICAL SPACE
  const BoxArray& grids = boxes();
  int ngrds = grids.size();
  const Real* dx = geom.CellSize();
  const Box& domain = geom.Domain();

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
    Vector<Real> &bloc = bcloc[face];
    Vector<RadBoundaryCondition> &bctag = bcond[face];

    int dir = face.coordDir();
    Real delta = dx[dir]*ratio[dir];
    int face_bc_type = (face.isLow() ? bc.lo(dir) : bc.hi(dir));

    for (int i = 0; i < ngrds; i++) {
      const Box& grd = grids[i];

      if (domain[face] == grd[face] && !geom.isPeriodic(dir)) {
/*
        // All physical bc values are located on face
        if (face_bc_type == amrex::BCType::ext_dir) {
          bctag[i] = AMREX_LO_DIRICHLET;
          bloc[i] = 0.;
        }
        else if (face_bc_type == EXTRAP || face_bc_type == amrex::BCType::hoextrap || face_bc_type == amrex::BCType::reflect_even) {
          bctag[i] = AMREX_LO_NEUMANN;
          bloc[i] = 0.;
        }
        else if (face_bc_type == amrex::BCType::reflect_odd) {
          bctag[i] = AMREX_LO_REFLECT_ODD;
          bloc[i] = 0.;
        }
*/
        if (face_bc_type == AMREX_LO_DIRICHLET   || face_bc_type == AMREX_LO_NEUMANN ||
            face_bc_type == AMREX_LO_REFLECT_ODD) {
          bctag[i] = face_bc_type;
          bloc[i] = 0.;
        }
        else if (face_bc_type == AMREX_LO_MARSHAK || face_bc_type == AMREX_LO_SANCHEZ_POMRANING) {
          bctag[i] = face_bc_type;
          //gives asymmetric, second-order version of Marshak b.c.
          // (worked for bbmg, works with nonsymmetric hypre solvers):
          bloc[i] = 0.;
          //gives symmetric version of Marshak b.c.
          //(hypre symmetric solvers ignore bloc and do this automatically):
          //bloc[i] = -0.5 * dx[dir];
        }
        else {
          std::cerr << "RadiativeBoundary---Not a recognized boundary condition" << std::endl;
          exit(1);
        }
      }
      else {
        // internal bndry
        bctag[i] = AMREX_LO_DIRICHLET;
        bloc[i] = 0.5*delta;
      }
    }
  }
}

void RadiativeBoundary::setBoundaryFluxConditions(const BCRec& bc, const BoundaryMode bc_value_mode)
{
  const BoxArray& grids = boxes();
//  int ngrds = grids.size();
  const Real* dx = geom.CellSize();
  const Real* xlo = geom.ProbLo();
  const Box& domain = geom.Domain();

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
//    Vector<Real> &bloc = bcloc[face];
//    Vector<RadBoundaryCondition> &bctag = bcond[face];

    int dir = face.coordDir();
    int face_bc_type = (face.isLow() ? bc.lo(dir) : bc.hi(dir));

    int face_bc_kind = (bc_value_mode == Nonzero_BC && !correction)
      ? bc_kind[face] : 0;
    Real value = (bc_value_mode == Nonzero_BC && !correction)
      ? bcval[face] : 0.0;

    for (FabSetIter bi(bndry[face]); bi.isValid(); ++bi) {
      int i = bi.index();
      const Box& grd = grids[i];

      if (domain[face] == grd[face] && !geom.isPeriodic(dir)) {
        if (bc_kind[face] <= 1) {
          if (face_bc_type == AMREX_LO_MARSHAK   || face_bc_type == AMREX_LO_SANCHEZ_POMRANING ||
              face_bc_type == AMREX_LO_DIRICHLET || face_bc_type == AMREX_LO_NEUMANN) {
              setValue(face, i, value);
          }
        }
        else {
          if(ParallelDescriptor::IOProcessor()) {
            std::cout << "RadiativeBoundary ERROR 2: feature not implemented" << std::endl;
          }
          exit(2);

#if 0
          FArrayBox& bnd_fab = bndry[face][bi];
          BaseFab<int>& bc_type_fab = bc_type_fabs[face][i];
          FORT_RADIATIVE_BOUNDARY_FILL(BL_TO_FORTRAN(bnd_fab),
                         bc_type_fab.dataPtr(),
                         AMREX_ARLIM(domain.loVect()), AMREX_ARLIM(domain.hiVect()), dx, xlo, time);
          if (face_bc_kind == 0) {
            // Homogeneous case.  We called RADBNDRY2 only to set bc_type_fab right.
            setValue(face, i, value);
          }
#endif
        }
      }
    }
  }
}

// *************************************************************************

void RadiativeBoundary::setHomogValues(const BCRec& bc, IntVect& ratio)
{

  setBoundaryConditions(bc, geom, ratio);

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
    for (FabSetIter bi(bndry[face]); bi.isValid(); ++bi) {
      FArrayBox& bnd_fab = bndry[face][bi];
      bnd_fab.setVal<RunOn::Host>(0.0);
    }
  }
}
