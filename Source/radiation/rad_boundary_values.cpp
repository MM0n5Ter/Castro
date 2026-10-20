#include <AMReX_ParmParse.H>
#include <AMReX_LO_BCTYPES.H>
#include <rad_boundary_values.H>

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
    if (!(face.isLow())) {
      bc_kind[face] = hi_bc_kind[dir];
      bcval[face]  = hi_bc_value[dir];
    } else {
      bc_kind[face] = lo_bc_kind[dir];
      bcval[face]  = lo_bc_value[dir];
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

void RadiativeBoundary::setBoundaryFluxConditions(const BCRec& bc, const BoundaryMode bc_value_mode)
{
  const BoxArray& grids = boxes();
  const Real* dx = geom.CellSize();
  const Real* xlo = geom.ProbLo();
  const Box& domain = geom.Domain();

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());

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
            setValue(face, i, value);
          }
#endif
        }
      }
    }
  }
}

void RadiativeBoundary::setBoundaryConditions(const BCRec& bc,
                             const Geometry& geom, IntVect& ratio)
{

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
        if (face_bc_type == AMREX_LO_DIRICHLET   || face_bc_type == AMREX_LO_NEUMANN ||
            face_bc_type == AMREX_LO_REFLECT_ODD) {
          bctag[i] = face_bc_type;
          bloc[i] = 0.;
        }
        else if (face_bc_type == AMREX_LO_MARSHAK || face_bc_type == AMREX_LO_SANCHEZ_POMRANING) {
          bctag[i] = face_bc_type;
          bloc[i] = 0.;
        }
        else {
          std::cerr << "RadiativeBoundary---Not a recognized boundary condition" << std::endl;
          exit(1);
        }
      }
      else {
        bctag[i] = AMREX_LO_DIRICHLET;
        bloc[i] = 0.5*delta;
      }
    }
  }
}

int         MultiGroupBoundary::ngroups = 1;
int         MultiGroupBoundary::first = 1;
Vector<int>  MultiGroupBoundary::bc_kind(2*AMREX_SPACEDIM);
Vector< Vector<Real> > MultiGroupBoundary::bcval(2*AMREX_SPACEDIM);
Real        MultiGroupBoundary::time = 0.0;
int         MultiGroupBoundary::correction = 0;

MultiGroupBoundary::MultiGroupBoundary(const BoxArray& _grids,
                       const DistributionMapping& _dmap,
                       const int _num_groups,
                       const Geometry& _geom) :
    BaseBoundary(_grids,_dmap,_num_groups,_geom)
{
  if (first)
    init(_num_groups);
  first = 0;

  const BoxArray& grids = boxes();
  const Box& domain = geom.Domain();

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
#if 0
          FORT_RADIATIVE_BOUNDARY_FILL(BL_TO_FORTRAN(bndry[face][bi]),
                         bc_type_fabs[face][igrid]->dataPtr(),
                         EXTENT_LIMITS(domain.loVect()), EXTENT_LIMITS(domain.hiVect()), dx, xlo, time);
#endif
        }
      }
    }
  }
}

void MultiGroupBoundary::init(const int _num_groups)
{

  ngroups = _num_groups;

  ParmParse pp("radiation");

  Vector<int> lo_bc_kind(AMREX_SPACEDIM, 0), hi_bc_kind(AMREX_SPACEDIM, 0);
  pp.queryarr("lo_bc_kind",lo_bc_kind,0,AMREX_SPACEDIM);
  pp.queryarr("hi_bc_kind",hi_bc_kind,0,AMREX_SPACEDIM);

  Vector< Vector<Real> > lo_bc_value(AMREX_SPACEDIM), hi_bc_value(AMREX_SPACEDIM);
  lo_bc_value[0].resize(ngroups, 0.0);
  hi_bc_value[0].resize(ngroups, 0.0);
  pp.queryarr("lo_bcval0", lo_bc_value[0], 0, ngroups);
  pp.queryarr("hi_bcval0", hi_bc_value[0], 0, ngroups);

#if (AMREX_SPACEDIM >= 2)
  lo_bc_value[1].resize(ngroups, 0.0);
  hi_bc_value[1].resize(ngroups, 0.0);
  pp.queryarr("lo_bcval1", lo_bc_value[1], 0, ngroups);
  pp.queryarr("hi_bcval1", hi_bc_value[1], 0, ngroups);
#endif

#if (AMREX_SPACEDIM == 3)
  lo_bc_value[2].resize(ngroups, 0.0);
  hi_bc_value[2].resize(ngroups, 0.0);
  pp.queryarr("lo_bcval2", lo_bc_value[2], 0, ngroups);
  pp.queryarr("hi_bcval2", hi_bc_value[2], 0, ngroups);
#endif

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
    int dir = face.coordDir();
    bcval[face].resize(ngroups);
    if (face.isLow()) {
      bc_kind[face] = lo_bc_kind[dir];
      for(int rad_group_idx = 0; rad_group_idx < ngroups; rad_group_idx++) {
        bcval[face][rad_group_idx] = lo_bc_value[dir][rad_group_idx];
      }
    }
    else {
      bc_kind[face] = hi_bc_kind[dir];
      for(int rad_group_idx = 0; rad_group_idx < ngroups; rad_group_idx++) {
        bcval[face][rad_group_idx] = hi_bc_value[dir][rad_group_idx];
      }
    }
  }
}

void MultiGroupBoundary::setHomogValues(const BCRec& bc, IntVect& ratio)
{

  setBoundaryConditions(bc, geom, ratio);

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());
    for (FabSetIter bi(bndry[face]); bi.isValid(); ++bi) {
      FArrayBox& bnd_fab = bndry[face][bi];
      bnd_fab.setVal<RunOn::Host>(0.);
    }
  }
}

void MultiGroupBoundary::setBoundaryFluxConditions(const BCRec& bc, const BoundaryMode bc_value_mode)
{
  const BoxArray& grids = boxes();
  const Real* dx = geom.CellSize();
  const Real* xlo = geom.ProbLo();
  const Box& domain = geom.Domain();

  Vector<Real> group_face_values;
  group_face_values.resize(ngroups);

  for (OrientationIter fi; fi; ++fi) {
    Orientation face(fi());

    int dir = face.coordDir();
    int face_bc_type = (face.isLow() ? bc.lo(dir) : bc.hi(dir));

    int face_bc_kind = (bc_value_mode == Nonzero_BC && !correction)
      ? bc_kind[face] : 0;
    if(bc_value_mode == Nonzero_BC && !correction) {
      for(int rad_group_idx = 0; rad_group_idx < ngroups; rad_group_idx++) {
        group_face_values[rad_group_idx] = bcval[face][rad_group_idx];
      }
    }
    else {
      for(int rad_group_idx = 0; rad_group_idx < ngroups; rad_group_idx++) {
        group_face_values[rad_group_idx] = 0.0;
      }
    }

    for (FabSetIter bi(bndry[face]); bi.isValid(); ++bi) {
      int i = bi.index();
      const Box& grd = grids[i];

      if (domain[face] == grd[face] && !geom.isPeriodic(dir)) {
        if (bc_kind[face] <= 1) {
          if (face_bc_type == AMREX_LO_MARSHAK   || face_bc_type == AMREX_LO_SANCHEZ_POMRANING ||
              face_bc_type == AMREX_LO_DIRICHLET || face_bc_type == AMREX_LO_NEUMANN) {
              for(int rad_group_idx = 0; rad_group_idx < ngroups; rad_group_idx++) {
                  bndry[face][bi].setVal<RunOn::Host>(group_face_values[rad_group_idx], rad_group_idx);
              }
          }
        }
        else {
          if(ParallelDescriptor::IOProcessor()) {
            std::cout << "MultiGroupBoundary ERROR 2: feature not implemented" << std::endl;
          }
          exit(2);

#if 0
          FArrayBox& bnd_fab = bndry[face][bi];
          BaseFab<int>& bc_type_fab = *(bc_type_fabs[face][i]);

          FORT_RADIATIVE_BOUNDARY_FILL(BL_TO_FORTRAN(bnd_fab),
                         bc_type_fab.dataPtr(), AMREX_ARLIM(domain.loVect()), AMREX_ARLIM(domain.hiVect()), dx, xlo, time);
#endif
          if (face_bc_kind == 0) {
          }
        }
      }
    }
  }
}

void MultiGroupBoundary::setBoundaryConditions(const BCRec& bc,
                             const Geometry& geom, IntVect& ratio)
{

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
        if (face_bc_type == AMREX_LO_DIRICHLET   || face_bc_type == AMREX_LO_NEUMANN ||
            face_bc_type == AMREX_LO_REFLECT_ODD) {
          bctag[i] = face_bc_type;
          bloc[i] = 0.;
        }
        else if (face_bc_type == AMREX_LO_MARSHAK || face_bc_type == AMREX_LO_SANCHEZ_POMRANING) {
          bctag[i] = face_bc_type;
          bloc[i] = 0.;
        }
        else {
          std::cerr << "MultiGroupBoundary---Not a recognized boundary condition" << std::endl;
          exit(1);
        }
      }
      else {
        bctag[i] = AMREX_LO_DIRICHLET;
        bloc[i] = 0.5*delta;
      }
    }
  }
}
