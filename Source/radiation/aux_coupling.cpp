#include <Radiation.H>

#include <AMReX_ParmParse.H>

#include <HypreSolvers.H>
#include <elliptic_face_ops.H>
#include <radiation_helpers.H>
#include <AMReX_LO_BCTYPES.H>

#include <_hypre_sstruct_mv.h>
#include <HYPRE_krylov.h>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

CoarseBoundaryAuxStore::CoarseBoundaryAuxStore(const BoxArray& _coarse_boxes,
                                 const DistributionMapping& _coarse_dmap,
                                 const BoxArray& _fine_boxes, Location loc)
    : BoundaryAuxBase(_coarse_dmap), coarse_boxes(_coarse_boxes), fgrids(_fine_boxes)
{
  BL_ASSERT(loc == OUTSIDE_GRID || loc == HALO);

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    aux[ori].resize(coarse_boxes.size());
    msk[ori].resize(coarse_boxes.size());

    fine_map[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      std::list<Box> bl;
      std::list<int> fi;
      for (int j = 0; j < fgrids.size(); j++) {
        Box face = amrex::adjCell(fgrids[j], ori);
        if (coarse_boxes[i].intersects(face)) {
          bl.push_back(face & coarse_boxes[i]);
          fi.push_back(j);
        }
      }


      int n = bl.size();
      aux[ori][i].resize(n);
      msk[ori][i].resize(n);

      fine_map[ori][i].resize(n);

      int j = 0;
      for (std::list<int>::iterator it = fi.begin(); it != fi.end(); ++it) {
        fine_map[ori][i][j++] = *it;
      }

      j = 0;
      for (std::list<Box>::iterator it = bl.begin(); it != bl.end(); ++it) {
        aux[ori][i][j].reset(new BoxAuxStorage(*it));
        Box mask_region = *it;
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
          if (dir == ori.coordDir())
            continue;
          mask_region.grow(dir,1);
        }
        msk[ori][i][j].reset(new Mask(mask_region));

        msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::outside_domain);


        for (int k = 0; k < coarse_boxes.size(); k++) {
          if (coarse_boxes[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::not_covered,
                                                  (coarse_boxes[k] & mask_region), 0);
          }
        }

        for (int k = 0; k < fgrids.size(); k++) {
          if (fgrids[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::covered,
                                                  (fgrids[k] & mask_region), 0);
          }
        }

        j++;
      }
    }
  }

  link_secondary_entries(loc);
}

CoarseBoundaryAuxStore::CoarseBoundaryAuxStore(const CoarseBoundaryAuxStore& other, Location loc)
    : BoundaryAuxBase(other.distributionMap), coarse_boxes(other.coarse_boxes), fgrids(other.fgrids)
{
  BL_ASSERT(loc == OUTSIDE_GRID || loc == HALO);

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    aux[ori].resize(coarse_boxes.size());
    msk[ori].resize(coarse_boxes.size());

    fine_map[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = other.aux[ori][i].size();

      aux[ori][i].resize(n);
      msk[ori][i].resize(n);

      fine_map[ori][i].resize(n);

      for (int j = 0; j < n; j++) {
        fine_map[ori][i][j] = other.fine_map[ori][i][j];

        aux[ori][i][j].reset(new BoxAuxStorage(other.aux[ori][i][j]->box()));

        msk[ori][i][j].reset(new Mask(other.msk[ori][i][j]->box()));
        msk[ori][i][j]->copy<RunOn::Host>(*other.msk[ori][i][j]);
      }
    }
  }

  link_secondary_entries(loc);
}

CoarseBoundaryAuxStore::CoarseBoundaryAuxStore(const BoxArray& _coarse_boxes,
                                 const BoxArray& _fine_boxes,
                                 const CoarseBoundaryAuxStore& other, Location loc)
    : BoundaryAuxBase(other.distributionMap), coarse_boxes(_coarse_boxes), fgrids(_fine_boxes)
{
  BL_ASSERT(loc == OUTSIDE_GRID || loc == HALO);

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    aux[ori].resize(coarse_boxes.size());
    msk[ori].resize(coarse_boxes.size());

    fine_map[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = other.aux[ori][i].size();

      aux[ori][i].resize(n);
      msk[ori][i].resize(n);

      fine_map[ori][i].resize(n);

      for (int j = 0; j < n; j++) {
        fine_map[ori][i][j] = other.fine_map[ori][i][j];

        Box face = amrex::adjCell(fgrids[fine_map[ori][i][j]], ori);
        aux[ori][i][j].reset(new BoxAuxStorage(face & coarse_boxes[i]));

        Box mask_region = aux[ori][i][j]->box();
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
          if (dir == ori.coordDir())
            continue;
          mask_region.grow(dir,1);
        }
        msk[ori][i][j].reset(new Mask(mask_region));

        msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::outside_domain);


        for (int k = 0; k < coarse_boxes.size(); k++) {
          if (coarse_boxes[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::not_covered,
                                                  (coarse_boxes[k] & mask_region), 0);
          }
        }

        for (int k = 0; k < fgrids.size(); k++) {
          if (fgrids[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::covered,
                                                  (fgrids[k] & mask_region), 0);
          }
        }

      }
    }
  }

  link_secondary_entries(loc);
}

void CoarseBoundaryAuxStore::rebuild_links(Location loc)
{
  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = aux[ori][i].size();
      for (int j = 0; j < n; j++) {
        const Box& reg = aux[ori][i][j]->box();
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
            (*aux[ori][i][j])(v).clear();
        }
      }
    }
  }

  link_secondary_entries(loc);
}

void CoarseBoundaryAuxStore::link_secondary_entries(Location loc)
{

  if (loc == OUTSIDE_GRID) {
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      for (OrientationIter outer_orient_iter; outer_orient_iter; ++outer_orient_iter) {
        Orientation om = outer_orient_iter();
        for (OrientationIter inner_orient_iter(om); ++inner_orient_iter; ) {
          Orientation os = inner_orient_iter();

          for (int jm = 0; jm < aux[om][i].size(); jm++) {
            const Box& bm = aux[om][i][jm]->box();
            for (int js = 0; js < aux[os][i].size(); js++) {
                const Box& bs = aux[os][i][js]->box();

              if (bm.intersects(bs)) {
                Box reg = (bm & bs);
                for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
                    (*aux[om][i][jm])(v).add_secondary_link(&(*aux[os][i][js])(v));
                }
              }

            }
          }

        }
      }
    }
  }
}

void CoarseBoundaryAuxStore::makeFaceStorage(IntVect& rat, int ncomp)
{
  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    face_fabs[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = aux[ori][i].size();
      face_fabs[ori][i].resize(n);
      for (int j = 0; j < n; j++) {
          Box face_box = aux[ori][i][j]->box();
        face_box.shiftHalf(ori.coordDir(), (ori.isLow() ? 1 : -1));
        face_box.refine(rat);
        face_fabs[ori][i][j].reset(new FArrayBox(face_box, ncomp));
      }
    }
  }
}

void CoarseBoundaryAuxStore::resetFaceStorage(IntVect& rat, int ncomp)
{
  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = face_fabs[ori][i].size();
      for (int j = 0; j < n; j++) {
        const Box& face_box = face_fabs[ori][i][j]->box();
        face_fabs[ori][i][j]->resize(face_box, ncomp);
      }
    }
  }
}

void CoarseBoundaryAuxStore::storeFaceCoeffs(const Orientation ori,
                                   MultiFab& src,
                                   int srccomp,
                                   int destcomp,
                                   int numcomp)
{
  MultiFabCopyDescriptor face_copier;

  MultiFabId mfid = face_copier.RegisterMultiFab(&src);

  Vector< Vector<FillBoxId> > fill_box_ids(face_fabs[ori].size());
  for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
    int n = face_fabs[ori][i].size();
    fill_box_ids[i].resize(n);
    for (int j = 0; j < n; j++) {
      fill_box_ids[i][j] = face_copier.AddBox(mfid, face_fabs[ori][i][j]->box(), NULL,
                               fine_map[ori][i][j],
                               srccomp, destcomp, numcomp);
    }
  }

  face_copier.CollectData();

  for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
    int n = face_fabs[ori][i].size();
    for (int j = 0; j < n; j++) {
      face_copier.FillFab(mfid, fill_box_ids[i][j], *face_fabs[ori][i][j]);
    }
  }
}

BoundaryAuxStore::BoundaryAuxStore(const BoxArray& _grids,
                         const DistributionMapping& _dmap,
                         Location loc)
    : BoundaryAuxBase(_dmap), grids(_grids)
{
  BL_ASSERT(loc == INTERIOR || loc == HALO);

  int normal_side_index = (loc == INTERIOR) ? 0 : 1;

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();
    aux[ori].resize(grids.size());
    int ishift = ori.isLow() ? 1-normal_side_index : normal_side_index-1;
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      Box b = amrex::adjCell(grids[i], ori);
      aux[ori][i].reset(new BoxAuxStorage(b.shift(ori.coordDir(), ishift)));
    }
  }


  if (loc == INTERIOR) {
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
#if 1
      for (OrientationIter outer_orient_iter; outer_orient_iter; ++outer_orient_iter) {
        Orientation om = outer_orient_iter();
        const Box& bm = aux[om][i]->box();
        for (OrientationIter inner_orient_iter; inner_orient_iter; ++inner_orient_iter) {
          Orientation os = inner_orient_iter();
          if (os.coordDir() > om.coordDir()) {
            const Box& bs = aux[os][i]->box();
            if (bm.intersects(bs)) {
              Box reg = (bm & bs);
              for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
                  (*aux[om][i])(v).add_secondary_link(&(*aux[os][i])(v));
              }
            }
          }
        }
      }
#elif 0
      Orientation face_x_low(0, Orientation::low);
      Orientation face_x_high(0, Orientation::high);
      Orientation face_y_low(1, Orientation::low);
      Orientation face_y_high(1, Orientation::high);
      IntVect p = aux[face_x_low][i]->box().smallEnd();
      (*aux[face_x_low][i])(p).add_secondary_link(&(*aux[face_y_low][i])(p));
      p = aux[face_x_low][i]->box().bigEnd();
      (*aux[face_x_low][i])(p).add_secondary_link(&(*aux[face_y_high][i])(p));
      p = aux[face_x_high][i]->box().smallEnd();
      (*aux[face_x_high][i])(p).add_secondary_link(&(*aux[face_y_low][i])(p));
      p = aux[face_x_high][i]->box().bigEnd();
      (*aux[face_x_high][i])(p).add_secondary_link(&(*aux[face_y_high][i])(p));
#elif 0
      for (OrientationIter outer_orient_iter; outer_orient_iter; ++outer_orient_iter) {
        Orientation om = outer_orient_iter();
        const Box& bm = aux[om][i]->box();
        for (OrientationIter inner_orient_iter(om); ++inner_orient_iter; ) {
          Orientation os = inner_orient_iter();
          const Box& bs = aux[os][i]->box();
          if (bm.intersects(bs)) {
            Box reg = (bm & bs);
            for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
                (*aux[om][i])(v).add_secondary_link(&(*aux[os][i])(v));
            }
          }
        }
      }
#endif
    }
  }
}

int BoundaryAuxBase::firstOwnedIndex()
{
  const int MyProc = ParallelDescriptor::MyProc();
  int i = 0;
  while (distributionMap[i] != MyProc) {
    i++;
    if (i >= distributionMap.size()) {
        i = -1;
        break;
    }
  }
  return i;
}

int BoundaryAuxBase::nextOwnedIndex(int i)
{
  const int MyProc = ParallelDescriptor::MyProc();
  do {
    i++;
    if (i >= distributionMap.size()) {
        i = -1;
        break;
    }
  } while (distributionMap[i] != MyProc);
  return i;
}

int AuxVar::read_stencil_sites(Vector<int>& levels, Vector<IntVect>& cells)
{
  if (secondary()) {
    return 1;
  }

  int n = a.size();
  levels.resize(n);
  cells.resize(n);

  int i = 0;
  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->other != NULL) {
      return 2;
    }
    levels[i] = it->level;
    cells[i]  = it->index;
    i++;
  }

  return 0;
}

int AuxVar::read_stencil_values(Vector<Real>& values)
{
  if (secondary()) {
    return 1;
  }

  int n = a.size();
  values.resize(n);

  int i = 0;
  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->other != NULL) {
      return 2;
    }
    values[i] = it->val;
    i++;
  }

  return 0;
}

void AuxVar::collapse()
{



  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ) {
    AuxVar *o = it->other;
    if (o) {
      std::list<AuxLink>::iterator kt = it;
      a.insert(++kt, o->a.begin(), o->a.end());

      for (std::list<AuxLink>::iterator jt = it; ++jt != kt; ) {
        jt->val *= it->val;
      }

      kt = it;
      ++it;
      a.erase(kt);
    }
    else {
      ++it;
    }
  }


  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ++it) {
    for (std::list<AuxLink>::iterator jt = it; ++jt != a.end(); ) {
      if (it->same_destination(*jt)) {
        it->val += jt->val;
        std::list<AuxLink>::iterator kt = jt;
        --jt;
        a.erase(kt);
      }
    }
  }
}

void AuxVar::clear()
{
  a.clear();
  secondary_pushed = 0;
}
