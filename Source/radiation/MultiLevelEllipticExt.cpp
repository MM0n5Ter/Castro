#include <HypreSolvers.H>
#include <elliptic_face_ops.H>
#include <AMReX_LO_BCTYPES.H>

#include <_hypre_sstruct_mv.h>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

ExtendedMultiLevelSolver::~ExtendedMultiLevelSolver()
{
}

void ExtendedMultiLevelSolver::setGradientCoefficients1(int level, const MultiFab &d1, int dir)
{
  BL_PROFILE("ExtendedMultiLevelSolver::setGradientCoefficients1");

  BL_ASSERT( d1.ok() );

  int ncomp=1;
  int ngrow=0;

  if (!gradient_coefs1[level]) {
    gradient_coefs1[level].reset(new Array<MultiFab, AMREX_SPACEDIM>);

    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      (*gradient_coefs1[level])[i].define(grids[level], dmap[level], ncomp, ngrow);
      (*gradient_coefs1[level])[i].setVal(0.0);
    }
  }

  BL_ASSERT( d1.boxArray() == (*gradient_coefs1[level])[dir].boxArray() );

  MultiFab::Copy((*gradient_coefs1[level])[dir], d1, 0, 0, ncomp, ngrow);
}

void ExtendedMultiLevelSolver::setGradientCoefficients2(int level, const MultiFab &d2, int dir)
{
  BL_PROFILE("ExtendedMultiLevelSolver::setGradientCoefficients2");

  BL_ASSERT( d2.ok() );

  int ncomp=1;
  int ngrow=0;

  if (!gradient_coefs2[level]) {
    gradient_coefs2[level].reset(new Array<MultiFab, AMREX_SPACEDIM>);

    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      BoxArray edge_boxes(grids[level]);
      edge_boxes.surroundingNodes(i);
      (*gradient_coefs2[level])[i].define(edge_boxes, dmap[level], ncomp, ngrow);
      (*gradient_coefs2[level])[i].setVal(0.0);
    }
  }

  BL_ASSERT( d2.boxArray() == (*gradient_coefs2[level])[dir].boxArray() );

  MultiFab::Copy((*gradient_coefs2[level])[dir], d2, 0, 0, ncomp, ngrow);
}

void ExtendedMultiLevelSolver::setAdvectionCoefficients(int level, const MultiFab &c, int dir)
{
  BL_PROFILE("ExtendedMultiLevelSolver::setAdvectionCoefficients");

  BL_ASSERT( c.ok() );

  int ncomp=2;
  int ngrow=0;

  if (!cell_coefs[level]) {
    cell_coefs[level].reset(new Array<MultiFab, AMREX_SPACEDIM>);

    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      BoxArray edge_boxes(grids[level]);
      edge_boxes.surroundingNodes(i);
      (*cell_coefs[level])[i].define(edge_boxes, dmap[level], ncomp, ngrow);
      (*cell_coefs[level])[i].setVal(0.0);
    }
  }

  BL_ASSERT( c.boxArray() == (*cell_coefs[level])[dir].boxArray() );

  MultiFab::Copy((*cell_coefs[level])[dir], c, 0, 0, ncomp, ngrow);
}

void ExtendedMultiLevelSolver::setAuxMassCoefficients(int level, const MultiFab &a2, int dir)
{
  BL_PROFILE("ExtendedMultiLevelSolver::setAuxMassCoefficients");

  BL_ASSERT( a2.ok() );

  int ncomp=1;
  int ngrow=0;

  if (!aux_mass_coefs[level]) {
    aux_mass_coefs[level].reset(new Array<MultiFab, AMREX_SPACEDIM>);

    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      BoxArray edge_boxes(grids[level]);
      edge_boxes.surroundingNodes(i);
      (*aux_mass_coefs[level])[i].define(edge_boxes, dmap[level], ncomp, ngrow);
      (*aux_mass_coefs[level])[i].setVal(0.0);
    }
  }

  BL_ASSERT( a2.boxArray() == (*aux_mass_coefs[level])[dir].boxArray() );

  MultiFab::Copy((*aux_mass_coefs[level])[dir], a2, 0, 0, ncomp, ngrow);
}

void
ExtendedMultiLevelSolver::ExtrapolateToFace(BoxAuxStorage& extrap_value, BoxAuxStorage& coarse_interp,
          const Mask& msk, const Box& reg,
          const IntVect& vin, int r, int bho, int fine_lvl)
{
  if (bho == 1) {
    Real face_weight_interp = 3.0 / ((1 + r) * (3 + r));
    Real face_weight_near = ( 1.5 * r) / (1 + r);
    Real face_weight_far = (-0.5 * r) / (3 + r);
    IntVect vi2 = 2 * vin;
    for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
      if (msk(v) == RadBoundaryData::not_covered) {
        extrap_value(v).push(&coarse_interp(v),    face_weight_interp);
        extrap_value(v).push(fine_lvl, v-vin, face_weight_near);
        extrap_value(v).push(fine_lvl, v-vi2, face_weight_far);
      }
    }
  }
  else {
    Real face_weight_interp = 1.0 / (1 + r);
    Real face_weight_near = r / (1.0 + r);
    for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
      if (msk(v) == RadBoundaryData::not_covered) {
        extrap_value(v).push(&coarse_interp(v),    face_weight_interp);
        extrap_value(v).push(fine_lvl, v-vin, face_weight_near);
      }
    }
  }
}

void ExtendedMultiLevelSolver::assembleMatrix()
{
  BL_PROFILE("ExtendedMultiLevelSolver::assembleMatrix");

  MultiLevelEllipticSolver::assembleMatrix();

  if (0 && verbose >= 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "In ExtendedMultiLevelSolver::assembleMatrix(), the multipliers are:" << std::endl;
    std::cout << "  ExtendedMultiLevelSolver::alpha2 = " << alpha2 << std::endl;
    std::cout << "  ExtendedMultiLevelSolver::gamma  = " << gamma  << std::endl;
    std::cout << "  ExtendedMultiLevelSolver::delta1 = " << delta1 << std::endl;
    std::cout << "  ExtendedMultiLevelSolver::delta2 = " << delta2 << std::endl;
  }

#if (AMREX_SPACEDIM == 1)
  int offsets[3][2] = {{ 0,  0},
                       {-1,  0},
                       { 1,  0}};
#elif (AMREX_SPACEDIM == 2)
  int offsets[5][2] = {{ 0,  0},
                       {-1,  0},
                       { 1,  0},
                       { 0, -1},
                       { 0,  1}};
#elif (AMREX_SPACEDIM == 3)
  int offsets[7][3] = {{ 0,  0,  0},
                       {-1,  0,  0},
                       { 1,  0,  0},
                       { 0, -1,  0},
                       { 0,  1,  0},
                       { 0,  0, -1},
                       { 0,  0,  1}};
#endif

  auto stencil_slot = [&] (const IntVect& d) {
    for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
      if (d == IntVect(offsets[k])) {
        return k;
      }
    }
    return -1;
  };

  const int size = 2 * AMREX_SPACEDIM + 1;

  int stencil_indices[size];

  for (int i = 0; i < size; i++) {
    stencil_indices[i] = i;
  }

  FArrayBox matfab;
  FArrayBox mat_scratch_fab;
  constexpr int lo_col[3] = {1, 3, 5};
  constexpr int hi_col[3] = {2, 4, 6};

  for (int level = crse_level; level <= fine_level; level++) {
    int part = level - crse_level;

    for (MFIter mfi(*acoefs[level]); mfi.isValid(); ++mfi) {
      int i = mfi.index();
      const Box &reg = grids[level][i];

      matfab.resize(reg,size);
      auto mat = matfab.array();
      Elixir mat_hold = matfab.elixir();

      matfab.setVal<RunOn::Device>(0.0);


      if (aux_mass_coefs[level]) {
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            const Real fac = 0.25_rt * alpha2;

            auto a2 = (*aux_mass_coefs[level])[idim][mfi].array();

            const int di = (idim == 0);
            const int dj = (idim == 1);
            const int dk = (idim == 2);
            const int lo_comp = lo_col[idim];
            const int hi_comp = hi_col[idim];

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                mat(i,j,k,0) += fac * (a2(i,j,k) + a2(i+di,j+dj,k+dk));
                mat(i,j,k,lo_comp) += fac * a2(i,j,k);
                mat(i,j,k,hi_comp) += fac * a2(i+di,j+dj,k+dk);
            });
        }
      }

      if (cell_coefs[level]) {
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            const Real fac = 0.5_rt * gamma / geom[level].CellSize(idim);

            auto c = (*cell_coefs[level])[idim][mfi].array();

            const int di = (idim == 0);
            const int dj = (idim == 1);
            const int dk = (idim == 2);
            const int lo_comp = lo_col[idim];
            const int hi_comp = hi_col[idim];

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                mat(i,j,k,0) += -fac * (c(i,j,k) - c(i+di,j+dj,k+dk));
                mat(i,j,k,lo_comp) += -fac * c(i,j,k);
                mat(i,j,k,hi_comp) += fac * c(i+di,j+dj,k+dk);
            });
        }
      }

      if (gradient_coefs1[level]) {
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            const Real fac = 0.5_rt * delta1 / geom[level].CellSize(idim);

            auto d1 = (*gradient_coefs1[level])[idim][mfi].array();

            const int di = (idim == 0);
            const int dj = (idim == 1);
            const int dk = (idim == 2);
            const int lo_comp = lo_col[idim];
            const int hi_comp = hi_col[idim];

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                mat(i,j,k,lo_comp) += -fac * d1(i,j,k);
                mat(i,j,k,hi_comp) += fac * d1(i,j,k);
            });
        }
      }

      if (gradient_coefs2[level]) {
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            const Real fac = 0.5_rt * delta2 / geom[level].CellSize(idim);

            auto d2 = (*gradient_coefs2[level])[idim][mfi].array();

            const int di = (idim == 0);
            const int dj = (idim == 1);
            const int dk = (idim == 2);
            const int lo_comp = lo_col[idim];
            const int hi_comp = hi_col[idim];

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                mat(i,j,k,0) += fac * (d2(i,j,k) - d2(i+di,j+dj,k+dk));
                mat(i,j,k,lo_comp) += -fac * d2(i,j,k);
                mat(i,j,k,hi_comp) += fac * d2(i+di,j+dj,k+dk);
            });
        }
      }

      Gpu::streamSynchronize();


      mat_scratch_fab.resize(reg);
      Real* mat_tmp = mat_scratch_fab.dataPtr();
      int volume = reg.numPts();
      for (int s = 0; s < size; s++) {
        for (int k = 0; k < volume; k++) {
          mat_tmp[k] = matfab.dataPtr()[s * volume + k];
        }
        HYPRE_SStructMatrixAddToBoxValues(A, part, loV(reg), hiV(reg), 0,
                                          1, &stencil_indices[s], mat_tmp);
      }
    }

    Gpu::streamSynchronize();



    BoundaryAuxStore extrap_value(grids[level], dmap[level], BoundaryAuxStore::HALO);
    BoundaryAuxStore entry( grids[level], dmap[level], BoundaryAuxStore::INTERIOR);
    if (level == crse_level) {
      normal_grad_store[level].reset(new BoundaryAuxStore(grids[level], dmap[level], BoundaryAuxStore::HALO));
    }


    const Box& domain = bd[level]->getDomain();
    for (OrientationIter oitr; oitr; oitr++) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir), ves;
      vin = (ori.isLow() ? -vin : vin);
      ves = (ori.isLow() ?  ves : vin);
      Real h    = geom[level].CellSize(idir);
      Real h2   = 0.5 * h;
      Real th2  = 1.5 * h;
      Real face_sign = (ori.isLow() ? -1.0 : 1.0);
      Real ofh  = face_sign / h;
      Real signed_half_inv_h = face_sign * 0.5 / h;
      const Real advect_fac = gamma * ofh;
      const int i0 = ori.isLow();
      const int i1 = ori.isHigh();
      for (int i = entry.firstOwnedIndex(); entry.isValid(i);
           i = entry.nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        if (grids[level][i][ori] == domain[ori] || level == crse_level) {
          RadBoundaryCondition     bct = bd[level]->bndryConds(ori)[i];
          const Real      &bcl = bd[level]->bndryLocs(ori)[i];
          const Mask      &msk = bd[level]->bndryMasks(ori,i);


          reg.shift(-vin);
          for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
            if (msk(v+vin) != RadBoundaryData::covered) {

              if (aux_mass_coefs[level]) {
                Real fac = alpha2 * (*aux_mass_coefs[level])[idir][i](v+ves);
                entry(ori,i)(v).push(level, v,     -0.25 * fac);
                entry(ori,i)(v).push(level, v+vin, -0.25 * fac);
              }
              if (cell_coefs[level]) {
                entry(ori,i)(v).push(level, v,
                                      -advect_fac * (*cell_coefs[level])[idir][i](v+ves,i0));
                entry(ori,i)(v).push(level, v+vin,
                                      -advect_fac * (*cell_coefs[level])[idir][i](v+ves,i1));
              }
              if (gradient_coefs1[level]) {
                Real fac = delta1 * signed_half_inv_h * (*gradient_coefs1[level])[idir][i](v);
                entry(ori,i)(v).push(level, v,      fac);
                entry(ori,i)(v).push(level, v+vin, -fac);
              }
              if (gradient_coefs2[level]) {
                Real fac = delta2 * signed_half_inv_h * (*gradient_coefs2[level])[idir][i](v+ves);
                entry(ori,i)(v).push(level, v,      fac);
                entry(ori,i)(v).push(level, v+vin, -fac);
              }

              if (reg[ori] == domain[ori] && bd[level]->isMixedFace(ori)) {
                bct = (*(bd[level]->boundaryTypes(ori)[i]))(v+vin);
              }
              if (bct == AMREX_LO_DIRICHLET) {
                if (bho == 1) {
                  extrap_value(ori,i)(v+vin).push(level, v,
                                            (bcl * th2) / (h * (bcl + h2)));
                  extrap_value(ori,i)(v+vin).push(level, v-vin,
                                            (-bcl * h2) / (h * (bcl + th2)));
                  (*normal_grad_store[level])(ori,i)(v+vin).push(level, v,
                                            (bcl - th2) / (h*(bcl + h2)));
                  (*normal_grad_store[level])(ori,i)(v+vin).push(level, v-vin,
                                            (h2 - bcl) / (h*(bcl + th2)));
                }
                else {
                  extrap_value(ori,i)(v+vin).push(level, v,
                                             bcl / (bcl + h2));
                  (*normal_grad_store[level])(ori,i)(v+vin).push(level, v,
                                            -1.0 / (bcl + h2));
                }
                if (aux_mass_coefs[level]) {
                  Real fac = 0.5 * alpha2 * (*aux_mass_coefs[level])[idir][i](v+ves);
                  entry(ori,i)(v).push(&extrap_value(ori,i)(v+vin), fac);
                }
                if (cell_coefs[level]) {



                  entry(ori,i)(v).push(level, v,
                                        advect_fac
                                        * (*cell_coefs[level])[idir][i](v+ves,i0));
                  entry(ori,i)(v).push(level, v,
                                        advect_fac * (bcl - h2) / (bcl + h2)
                                        * (*cell_coefs[level])[idir][i](v+ves,i1));
                }
                if (gradient_coefs1[level]) {
                  Real fac = 0.5*delta1*face_sign * (*gradient_coefs1[level])[idir][i](v);
                  entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin), fac);
                }
                if (gradient_coefs2[level]) {
                  Real fac = 0.5*delta2*face_sign * (*gradient_coefs2[level])[idir][i](v+ves);
                  entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin), fac);
                }
              }
              else if (bct == AMREX_LO_NEUMANN) {
              }
              else if (bct == AMREX_LO_MARSHAK || bct == AMREX_LO_SANCHEZ_POMRANING) {
                if (bho == 1) {
                  extrap_value(ori,i)(v+vin).push(level, v,      1.5);
                  extrap_value(ori,i)(v+vin).push(level, v-vin, -0.5);
                }
                else {
                  extrap_value(ori,i)(v+vin).push(level, v,      1.0);
                }
                Real speed_factor = flux_scale;
                Real xi = 0.0;
                Real tmp = 0.0;
                Real fac = 0.5 * (tmp - 0.5 * (delta1 + delta2) * face_sign * xi);
                entry(ori,i)(v).push(&extrap_value(ori,i)(v+vin), speed_factor * fac);
              }
              else {
                amrex::Error("ExtendedMultiLevelSolver: unsupported boundary type");
              }
            }
          }
        }
      }
    }


    if (level > crse_level) {


      IntVect rat = fine_ratio[level-1];
      for (OrientationIter oitr; oitr; ++oitr) {
        Orientation ori = oitr();
        int idir = ori.coordDir();
        IntVect vin = amrex::BASISV(idir), ves;
        vin = (ori.isLow() ? -vin : vin);
        ves = (ori.isLow() ?  ves : vin);
        Real h = geom[level].CellSize(idir);
        Real face_sign = (ori.isLow() ? -1.0 : 1.0);
        Real signed_half_inv_h = face_sign * 0.5 / h;
        for (int i = coarse_interp[level]->firstOwnedIndex(); coarse_interp[level]->isValid(i);
             i = coarse_interp[level]->nextOwnedIndex(i)) {
          Box reg = amrex::adjCell(grids[level][i], ori);
          const Mask &msk = bd[level]->bndryMasks(ori,i);
          ExtrapolateToFace(extrap_value(ori,i),
                    (*coarse_interp[level])(ori,i),
                    msk, reg, vin, rat[idir], bho, level);
          reg.shift(-vin);
          for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
            if (msk(v+vin) == RadBoundaryData::not_covered) {
              if (aux_mass_coefs[level]) {
                Real fac = alpha2 * (*aux_mass_coefs[level])[idir][i](v+ves);
                entry(ori,i)(v).push(level, v,     -0.25 * fac);
                entry(ori,i)(v).push(level, v+vin, -0.25 * fac);
                entry(ori,i)(v).push(&extrap_value(ori,i)(v+vin), 0.5 * fac);
              }
              if (cell_coefs[level]) {
                BL_ASSERT(level == crse_level);

                Real fac = gamma * signed_half_inv_h * (*cell_coefs[level])[idir][i](v+ves);
                entry(ori,i)(v).push(level, v,     -fac);
                entry(ori,i)(v).push(level, v+vin, -fac);
                entry(ori,i)(v).push(&extrap_value(ori,i)(v+vin), 2.0 * fac);
              }
              if (gradient_coefs1[level]) {
                Real fac = delta1 * signed_half_inv_h * (*gradient_coefs1[level])[idir][i](v);
                entry(ori,i)(v).push(level, v,      fac);
                entry(ori,i)(v).push(level, v+vin, -fac);
                entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin), h * fac);
              }
              if (gradient_coefs2[level]) {
                Real fac = delta2 * signed_half_inv_h * (*gradient_coefs2[level])[idir][i](v+ves);
                entry(ori,i)(v).push(level, v,      fac);
                entry(ori,i)(v).push(level, v+vin, -fac);
                entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin), h * fac);
              }
            }
          }
        }
      }
    }


    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin);
      for (int i = entry.firstOwnedIndex(); entry.isValid(i);
           i = entry.nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        reg.shift(-vin);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
          if (!entry(ori,i)(v).empty() &&
              !entry(ori,i)(v).secondary()) {
            entry(ori,i)(v).collapse();
            Vector<int> levels;
            Vector<IntVect> cells;
            int retval = entry(ori,i)(v).read_stencil_sites(levels, cells);
            BL_ASSERT(retval == 0);
            Vector<Real> values;
            retval = entry(ori,i)(v).read_stencil_values(values);
            BL_ASSERT(retval == 0);
            int ientry = 2 * AMREX_SPACEDIM + 1;
            for (int j = 0; j < levels.size(); j++) {
              int needs_separate_entry = 1;
              if (levels[j] == level) {
                IntVect d = cells[j] - v;
                int k = stencil_slot(d);
                if (k >= 0) {
                  needs_separate_entry = 0;
                  HYPRE_SStructMatrixAddToValues(A, part, cellIndex1(v), 0,
                                                 1, &k, &values[j]);
                }
              }
              if (needs_separate_entry) {
                HYPRE_SStructMatrixAddToValues(A, part, cellIndex1(v), 0,
                                               1, &ientry, &values[j]);
                ientry++;
              }
            }
          }
        }
      }
    }

    if (level == crse_level) {
        normal_grad_store[level].reset();
        continue;
    }


    IntVect rat = fine_ratio[level-1];
    const BoxArray& ff_boxes(grids[level]);
    const BoxArray& cc_boxes(grids[level-1]);
    BoxArray fc_boxes(cc_boxes);
    fc_boxes.refine(rat);
    BoxArray cf_boxes(ff_boxes);
    cf_boxes.coarsen(rat);
    CoarseBoundaryAuxStore c_extrap_value(*c_extrap_deriv[level], BoundaryAuxStore::HALO);

    int nc = 0, pc[3];
    if (aux_mass_coefs[level]) {
      pc[0] = nc++;
    }
    if (cell_coefs[level]) {
      BL_ASSERT(level == crse_level);

      pc[1] = nc++;
    }
    if (gradient_coefs2[level]) {
      pc[2] = nc++;
    }
    c_connections[level]->rebuild_links(BoundaryAuxStore::OUTSIDE_GRID);
    c_connections[level]->resetFaceStorage(rat, nc);

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      if (aux_mass_coefs[level]) {
        c_connections[level]->storeFaceCoeffs(ori, (*aux_mass_coefs[level])[idir], 0, pc[0], 1);
      }
      if (cell_coefs[level]) {
        BL_ASSERT(level == crse_level);

        c_connections[level]->storeFaceCoeffs(ori,  (*cell_coefs[level])[idir], 0, pc[1], 1);
      }
      if (gradient_coefs2[level]) {
        c_connections[level]->storeFaceCoeffs(ori, (*gradient_coefs2[level])[idir], 0, pc[2], 1);
      }
      IntVect vin = amrex::BASISV(idir), ves;
      vin = (ori.isLow() ? -vin : vin);
      ves = (ori.isLow() ? -vin : ves);
      Real hc = geom[level-1].CellSize(idir);
      Real face_sign = (ori.isLow() ? -1.0 : 1.0);
      Real signed_inv_coarse_h = face_sign / hc;
      Real tang_avg_scale = 1.0;
      IntVect ve;
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      ve += (rat[jdir] - 1) * amrex::BASISV(jdir);
      tang_avg_scale /= rat[jdir];
#endif
#if (AMREX_SPACEDIM == 3)
      int kdir = (idir + 2) % 3;
      ve += (rat[kdir] - 1) * amrex::BASISV(kdir);
      tang_avg_scale /= rat[kdir];
#endif
      for (int i = c_connections[level]->firstOwnedIndex(); c_connections[level]->isValid(i);
           i = c_connections[level]->nextOwnedIndex(i)) {
        for (int j = 0; j < (*c_connections[level]).size(ori,i); j++) {
          const Box& reg = (*c_interp[level])(ori,i,j).box();
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box();
          const Mask& msk = (*c_interp[level]).mask(ori,i,j);
          ExtrapolateToFace(c_extrap_value(ori,i,j),
                    (*c_interp[level])(ori,i,j),
                    msk, reg, vin, rat[idir], bho, level);
          const FArrayBox& face_coefs = c_connections[level]->faceFab(ori,i,j);
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir);
            Box face(vf, vf + ve);
            if (msk(vf) == RadBoundaryData::not_covered) {
                (*c_connections[level])(ori,i,j)(vc).push(-1, vc-vin, 0.0);
              if (aux_mass_coefs[level]) {
                  (*c_connections[level])(ori,i,j)(vc)
                  .push(level-1, vc,
                        -0.25 * alpha2 * (*aux_mass_coefs[level-1])[idir][i](vc+ves));
                for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                    (*c_connections[level])(ori,i,j)(vc)
                        .push(&((c_extrap_value(ori,i,j))(v)),
                              0.5 * alpha2 * tang_avg_scale * face_coefs(v+ves, pc[0]));
                }
              }
              if (cell_coefs[level]) {
                  (*c_connections[level])(ori,i,j)(vc)
                      .push(level-1, vc,
                            0.5 * gamma * signed_inv_coarse_h * (*cell_coefs[level-1])[idir][i](vc+ves));
                for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                    (*c_connections[level])(ori,i,j)(vc)
                        .push(&((c_extrap_value(ori,i,j))(v)),
                              -gamma * signed_inv_coarse_h * tang_avg_scale * face_coefs(v+ves, pc[1]));
                }
              }
              if (gradient_coefs1[level]) {
                Real grad_coef = (*gradient_coefs1[level-1])[idir][i](vc);
                (*c_connections[level])(ori,i,j)(vc)
                  .push(level-1, vc,
                        -0.5 * delta1 * signed_inv_coarse_h * grad_coef);
                for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                  (*c_connections[level])(ori,i,j)(vc)
                      .push(&(*c_extrap_deriv[level])(ori,i,j)(v),
                              0.5 * delta1 * face_sign * tang_avg_scale * grad_coef);
                }
              }
              if (gradient_coefs2[level]) {
                (*c_connections[level])(ori,i,j)(vc)
                  .push(level-1, vc,
                        -0.5 * delta2 * signed_inv_coarse_h * (*gradient_coefs2[level-1])[idir][i](vc+ves));
                for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                  (*c_connections[level])(ori,i,j)(vc)
                      .push(&(*c_extrap_deriv[level])(ori,i,j)(v),
                      0.5 * delta2 * face_sign * tang_avg_scale * face_coefs(v+ves, pc[2]));
                }
              }
            }
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      for (int i = c_connections[level]->firstOwnedIndex(); c_connections[level]->isValid(i);
           i = c_connections[level]->nextOwnedIndex(i)) {
        for (int j = 0; j < (*c_connections[level]).size(ori,i); j++) {
          const Box& reg = (*c_interp[level])(ori,i,j).box();
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box();
          const Mask &msk = (*c_interp[level]).mask(ori,i,j);
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir);
            if (msk(vf) == RadBoundaryData::not_covered &&
                !(*c_connections[level])(ori,i,j)(vc).secondary()) {
              (*c_connections[level])(ori,i,j)(vc).collapse();
              Vector<int> levels;
              Vector<IntVect> cells;
              int retval = (*c_connections[level])(ori,i,j)(vc)
                .read_stencil_sites(levels, cells);
              BL_ASSERT(retval == 0);
              Vector<Real> values;
              retval = (*c_connections[level])(ori,i,j)(vc).read_stencil_values(values);
              BL_ASSERT(retval == 0);
              int ientry = 2 * AMREX_SPACEDIM + 1;
              for (int jj = 0; jj < levels.size(); jj++) {
                int needs_separate_entry = 1;
                if (levels[jj] == -1) {
                  IntVect d = cells[jj] - vc;
                  int k = stencil_slot(d);
                  if (k >= 0) {
                    needs_separate_entry = 0;
                    HYPRE_SStructMatrixSetValues(A, part-1, cellIndex1(vc), 0,
                                                 1, &k, &values[jj]);
                    BL_ASSERT(values[jj] == 0.0);
                  }
                  BL_ASSERT(needs_separate_entry == 0);
                }
                if (levels[jj] == level-1) {
                  IntVect d = cells[jj] - vc;
                  int k = stencil_slot(d);
                  if (k >= 0) {
                    needs_separate_entry = 0;
                    HYPRE_SStructMatrixAddToValues(A, part-1, cellIndex1(vc), 0,
                                                   1, &k, &values[jj]);
                  }
                }
                if (needs_separate_entry) {
                  HYPRE_SStructMatrixAddToValues(A, part-1, cellIndex1(vc), 0,
                                                 1, &ientry, &values[jj]);
                  ientry++;
                }
              }
            }
          }
        }
      }
    }
  }
}

void ExtendedMultiLevelSolver::fillLevelRhs(int level,
                                         MultiFab& rhs,
                                         BoundaryMode bc_variant)
{
  {
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "In ExtendedMultiLevelSolver::fillLevelRhs(), the multipliers are:"
           << std::endl;
      std::cout << "  ExtendedMultiLevelSolver::alpha  = " << alpha  << std::endl;
      std::cout << "  ExtendedMultiLevelSolver::alpha2 = " << alpha2 << std::endl;
      std::cout << "  ExtendedMultiLevelSolver::beta   = " << beta   << std::endl;
      std::cout << "  ExtendedMultiLevelSolver::gamma  = " << gamma  << std::endl;
      std::cout << "  ExtendedMultiLevelSolver::delta1 = " << delta1 << std::endl;
      std::cout << "  ExtendedMultiLevelSolver::delta2 = " << delta2 << std::endl;
    }

  }

  int part = level - crse_level;

  FArrayBox scratch_fab;
  for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    FArrayBox *f;
    if (rhs.nGrow() == 0) {
      f = &rhs[mfi];
    }
    else {
      f = &scratch_fab;
      f->resize(reg);
      f->copy<RunOn::Host>(rhs[mfi]);
    }
    Real* vec = f->dataPtr();


    HYPRE_SStructVectorSetBoxValues(b, part, loV(reg), hiV(reg), 0, vec);
  }

  if (!bc_variant) {
    return;
  }


  const Box& domain = bd[level]->getDomain();
  for (OrientationIter oitr; oitr; oitr++) {
    Orientation ori = oitr();
    int idir = ori.coordDir();
    IntVect vin = amrex::BASISV(idir), ves;
    vin = (ori.isLow() ? -vin : vin);
    ves = (ori.isLow() ?  ves : vin);
    Real h    = geom[level].CellSize(idir);
    Real h2   = 0.5 * h;
    Real th2  = 1.5 * h;
    Real face_sign = (ori.isLow() ? -1.0 : 1.0);
    Real ofh  = face_sign / h;
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
      int i = mfi.index();
      Box reg = amrex::adjCell(grids[level][i], ori);
      if (!(grids[level][i][ori] == domain[ori] || level == crse_level)) {
        continue;
      }

      RadBoundaryCondition     bct = bd[level]->bndryConds(ori)[i];
      const Real      &bcl = bd[level]->bndryLocs(ori)[i];
      const FArrayBox &fs  = bd[level]->bndryValues(ori)[mfi];
      const Mask      &msk = bd[level]->bndryMasks(ori,i);


      reg.shift(-vin);
      for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
        if (msk(v+vin) == RadBoundaryData::covered) {
          continue;
        }
        if (reg[ori] == domain[ori] && bd[level]->isMixedFace(ori)) {
          bct = (*(bd[level]->boundaryTypes(ori)[i]))(v+vin);
        }
        if (bct == AMREX_LO_DIRICHLET) {
          Real dirichlet_scale, dirichlet_weight;
          if (bho == 1) {
            dirichlet_scale = 1.0 / ((bcl + h2) * (bcl + th2));
            dirichlet_weight = h2 * th2 * dirichlet_scale;
            dirichlet_scale *= 2.0 * h;
          }
          else {
            dirichlet_scale = 1.0 / (bcl + h2);
            dirichlet_weight = h2 * dirichlet_scale;
          }
          if (bcoefs[level]) {
            Real tmp = ((dirichlet_scale / h) * fs(v+vin,bc_component) *
                        beta * (*bcoefs[level])[idir][mfi](v+ves));
            HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
          }
          if (aux_mass_coefs[level]) {
            Real tmp = ((-0.5 * dirichlet_weight) * fs(v+vin,bc_component) *
                        alpha2 * (*aux_mass_coefs[level])[idir][mfi](v+ves));
            HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
          }
          if (cell_coefs[level]) {



            Real fac = gamma * ofh;
            int  i0  = ori.isLow();
            Real tmp = (-fac * (2.0 * h2) / (bcl + h2) * fs(v+vin,bc_component)
                        * (*cell_coefs[level])[idir][mfi](v+ves,i0));

            HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
          }
          if (gradient_coefs1[level]) {
            Real tmp = ((-0.5 * dirichlet_scale) * fs(v+vin,bc_component) *
                        delta1 * face_sign * (*gradient_coefs1[level])[idir][mfi](v));
            HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
          }
          if (gradient_coefs2[level]) {
            Real tmp = ((-0.5 * dirichlet_scale) * fs(v+vin,bc_component) *
                        delta2 * face_sign * (*gradient_coefs2[level])[idir][mfi](v+ves));
            HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
          }
        }
        else if (bct == AMREX_LO_NEUMANN) {
          Real speed_factor = 1.0;
          Real xi = 0.0;
          Real tmp = beta / h;
          tmp = ((tmp - 0.5 * (delta1 + delta2) * face_sign * xi) *
                 speed_factor * fs(v+vin,bc_component));
          HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
        }
        else if (bct == AMREX_LO_MARSHAK || bct == AMREX_LO_SANCHEZ_POMRANING) {
          Real speed_factor = 1.0;
          Real xi = 0.0;
          Real tmp = beta / h;
          tmp = 2.0 * ((tmp - 0.5 * (delta1 + delta2) * face_sign * xi) *
                       speed_factor * fs(v+vin,bc_component));
          HYPRE_SStructVectorAddToValues(b, part, cellIndex1(v), 0, &tmp);
        }
      }
    }
  }
}

void ExtendedMultiLevelSolver::fillLevelVectors(int level,
                                         MultiFab& dest,
                                         int icomp,
                                         MultiFab& rhs,
                                         BoundaryMode bc_variant)
{
  BL_PROFILE("ExtendedMultiLevelSolver::fillLevelVectors");

  loadLevelSolution(level, dest, icomp);
  fillLevelRhs(level, rhs, bc_variant);
}

void ExtendedMultiLevelSolver::computeBoundaryTerm(int level,
                                      MultiFab* diffusive_face_term,
                                      MultiFab& sol_field,
                                      int icomp)
{
  BL_PROFILE("ExtendedMultiLevelSolver::computeBoundaryTerm");

  const Box& domain = bd[level]->getDomain();
#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(sol_field); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];
    for (OrientationIter oitr; oitr; oitr++) {
      int cdir(oitr());
      int idim = oitr().coordDir();
      const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
      const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
      const FArrayBox       &bcv  = bd[level]->bndryValues(oitr())[mfi];
      const Mask      &msk = bd[level]->bndryMasks(oitr(), i);

      if (reg[oitr()] == domain[oitr()]) {
        Array4<int const> bc_type_arr;
        int bctype = bct;
        if (bd[level]->isMixedFace(oitr())) {
          const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
          bc_type_arr = tf.array();
          bctype = -1;
        }
        ELLIPTIC_OPS::face_bc_term_3d(diffusive_face_term[idim][mfi].array(),
                       sol_field[mfi].array(icomp),
                       reg,
                       cdir, bctype, bc_type_arr, bcl,
                       bcv.array(bc_component),
                       msk.array(),
                       (*gradient_coefs2[level])[idim][mfi].array(),
                       geom[level].CellSize());
      }
      else {
          ELLIPTIC_OPS::face_bc_term(diffusive_face_term[idim][mfi].array(),
                        sol_field[mfi].array(icomp),
                        reg,
                        cdir, bct, bcl,
                        bcv.array(bc_component),
                        msk.array(),
                        (*gradient_coefs2[level])[idim][mfi].array(),
                        geom[level].CellSize());
      }
    }
  }
}
