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

void MultiLevelEllipticSolver::assembleMatrix()
{
  BL_PROFILE("MultiLevelEllipticSolver::assembleMatrix");

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

  const int size = 2 * AMREX_SPACEDIM + 1;
  int i, idim;

  int stencil_indices[size];

  for (i = 0; i < size; i++) {
    stencil_indices[i] = i;
  }

  BaseFab<GpuArray<Real, size>> matfab;
  FArrayBox sub_matrix_buf;
  for (int level = crse_level; level <= fine_level; level++) {
    int part = level - crse_level;

    for (MFIter mfi(*acoefs[level]); mfi.isValid(); ++mfi) {
      i = mfi.index();
      const Box &reg = grids[level][i];

      matfab.resize(reg);
      Real* mat = (Real*) matfab.dataPtr();
      Elixir mat_hold = matfab.elixir();

      hmac(reg, matfab.array(), (*acoefs[level])[mfi].array(), alpha);

      for (idim = 0; idim < AMREX_SPACEDIM; idim++) {
          fillBoundaryCoeffs(reg, matfab.array(), (*bcoefs[level])[idim][mfi].array(),
               beta, geom[level].CellSize(), idim);
      }

      const Box& domain = bd[level]->getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
        const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
        const Mask      &msk = bd[level]->bndryMasks(oitr(),i);
        const Box &bbox = (*bcoefs[level])[idim][mfi].box();
        const Box &msb  = msk.box();
        if (reg[oitr()] == domain[oitr()] || level == crse_level) {

          if (reg[oitr()] == domain[oitr()]) {
            Array4<const int> tfp{};
            int bctype = bct;
            if (bd[level]->isMixedFace(oitr())) {
              const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
              tfp = tf.array();
              bctype = -1;
            }
            const FArrayBox &fs = bd[level]->bndryValues(oitr())[mfi];
            Array4<const Real> opt_bc_alpha_arr{};
            if (SPa[level]) {
                opt_bc_alpha_arr = (*SPa[level])[mfi].array();
            }
            fillLevelMat3(reg, oitr().isLow(), idim, matfab.array(), cdir, bctype,
                   tfp, bho, bcl, msk.array(), (*bcoefs[level])[idim][mfi].array(),
                   beta, geom[level].CellSize(), geom[level].data(),
                   flux_scale, opt_bc_alpha_arr);
          }
          else {
              fillLevelMat(reg, matfab.array(), cdir, bct, bho, bcl,
                    msk.array(), (*bcoefs[level])[idim][mfi].array(),
                    beta, geom[level].CellSize());
          }
        }
        else {

          const RadBoundaryCondition coarse_boundary_type = AMREX_LO_NEUMANN;
          fillLevelMat(reg, matfab.array(), cdir, coarse_boundary_type, bho, bcl,
                msk.array(), (*bcoefs[level])[idim][mfi].array(),
                beta, geom[level].CellSize());
        }
      }

      if (subgrids[level][i].size() > 0) {
        for (int j = 0; j < subgrids[level][i].size(); j++) {
          const Box& sreg = subgrids[level][i][j];
          sub_matrix_buf.resize(sreg,size);
          Real* smat = sub_matrix_buf.dataPtr();
          for (IntVect v = sreg.smallEnd(); v <= sreg.bigEnd(); sreg.next(v)) {
            int is = sreg.index(v);
            int ir =  reg.index(v);
            for (int s = 0; s < size; s++) {
              smat[is * size + s] = mat[ir * size + s];
            }
          }
          HYPRE_SStructMatrixSetBoxValues(A, part, loV(sreg), hiV(sreg), 0,
                                          size, stencil_indices, smat);
        }
      }
      else {
        HYPRE_SStructMatrixSetBoxValues(A, part, loV(reg), hiV(reg), 0,
                                        size, stencil_indices, mat);
      }
    }

    if (level == crse_level) {
      continue;
    }

    BoundaryAuxStore entry(grids[level], dmap[level], BoundaryAuxStore::INTERIOR);
    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir), ves;
      vin = (ori.isLow() ? -vin : vin);
      ves = (ori.isLow() ?  ves : vin);
      Real h = geom[level].CellSize(idir);
      Real fine_div_factor = (-beta / h);
      for (int i = coarse_interp[level]->firstOwnedIndex(); coarse_interp[level]->isValid(i);
           i = coarse_interp[level]->nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        reg.shift(-vin);
        const Mask &msk = bd[level]->bndryMasks(ori,i);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
          if (msk(v+vin) == RadBoundaryData::not_covered) {
              entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin),
                                    fine_div_factor * (*bcoefs[level])[idir][i](v+ves));
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin);
      for (int i = coarse_interp[level]->firstOwnedIndex(); coarse_interp[level]->isValid(i);
           i = coarse_interp[level]->nextOwnedIndex(i)) {
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
                for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                  if (d == IntVect(offsets[k])) {
                    needs_separate_entry = 0;
                    HYPRE_SStructMatrixAddToValues(A, part, cellIndex1(v), 0,
                                                   1, &k, &values[j]);
                  }
                }
              }
              if (needs_separate_entry) {
                HYPRE_SStructMatrixSetValues(A, part, cellIndex1(v), 0,
                                             1, &ientry, &values[j]);
                ientry++;
              }
            }
          }
        }
      }
    }

    IntVect rat = fine_ratio[level-1];
    const BoxArray& ff_boxes(grids[level]);
    const BoxArray& cc_boxes(grids[level-1]);
    BoxArray fc_boxes(cc_boxes);
    fc_boxes.refine(rat);
    BoxArray cf_boxes(ff_boxes);
    cf_boxes.coarsen(rat);
    c_connections[level]->rebuild_links(BoundaryAuxStore::OUTSIDE_GRID);
    c_connections[level]->makeFaceStorage(rat);

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      c_connections[level]->storeFaceCoeffs(ori, (*bcoefs[level])[idir], 0, 0, 1);
      IntVect vin = amrex::BASISV(idir), ves;
      vin = (ori.isLow() ? -vin : vin);
      ves = (ori.isLow() ? -vin : ves);
      Real hc = geom[level-1].CellSize(idir);
      Real coarse_div_factor = (beta / hc);
      Real covered_div_factor = (-coarse_div_factor / hc);
      IntVect ve;
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      ve += (rat[jdir] - 1) * amrex::BASISV(jdir);
      coarse_div_factor /= rat[jdir];
#endif
#if (AMREX_SPACEDIM == 3)
      int kdir = (idir + 2) % 3;
      ve += (rat[kdir] - 1) * amrex::BASISV(kdir);
      coarse_div_factor /= rat[kdir];
#endif
      for (int i = c_connections[level]->firstOwnedIndex(); c_connections[level]->isValid(i);
           i = c_connections[level]->nextOwnedIndex(i)) {
          for (int j = 0; j < (*c_connections[level]).size(ori,i); j++) {
          const Box& reg = (*c_interp[level])(ori,i,j).box();
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box();
          const Mask& msk = c_interp[level]->mask(ori,i,j);
          const FArrayBox& face_coef_data = c_connections[level]->faceFab(ori,i,j);
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir);
            Box face(vf, vf + ve);
            if (msk(vf) == RadBoundaryData::not_covered) {
              (*c_connections[level])(ori,i,j)(vc).push(-1, vc-vin, 0.0);
              (*c_connections[level])(ori,i,j)(vc).push(level-1, vc,
                                                       covered_div_factor * (*bcoefs[level-1])[idir][i](vc+ves));
              for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                  (*c_connections[level])(ori,i,j)(vc)
                      .push(&(*c_extrap_deriv[level])(ori,i,j)(v), coarse_div_factor * face_coef_data(v+ves));
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
          const Mask &msk = c_interp[level]->mask(ori,i,j);
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
                  for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                    if (d == IntVect(offsets[k])) {
                      needs_separate_entry = 0;
                      HYPRE_SStructMatrixSetValues(A, part-1, cellIndex1(vc), 0,
                                                   1, &k, &values[jj]);
                      BL_ASSERT(values[jj] == 0.0);
                    }
                  }
                  BL_ASSERT(needs_separate_entry == 0);
                }
                if (levels[jj] == level-1) {
                  IntVect d = cells[jj] - vc;
                  for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                    if (d == IntVect(offsets[k])) {
                      needs_separate_entry = 0;
                      HYPRE_SStructMatrixAddToValues(A, part-1, cellIndex1(vc), 0,
                                                     1, &k, &values[jj]);
                    }
                  }
                }
                if (needs_separate_entry) {
                  HYPRE_SStructMatrixSetValues(A, part-1, cellIndex1(vc), 0,
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

void
MultiLevelEllipticSolver::fillLevelMat (const Box& bx,
                       Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                       int cdir, int bct, int bho, Real bcl,
                       Array4<int const> const& mask,
                       Array4<Real const> const& b,
                       Real beta, const Real* dx)
{
    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h;

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    const Real fac = beta / (h * h);

    Real bfm, bfv;
    Real stencil_weight_second, h2, th2;

    if (bct == AMREX_LO_DIRICHLET) {

        if (bho >= 1) {

            h2 = 0.5e0_rt * h;
            th2 = 3.e0_rt * h2;
            bfm = fac * (th2 - bcl) / (bcl + h2) - fac;
            stencil_weight_second = fac * (bcl - h2) / (bcl + th2);

        }
        else {

            bfv = (beta / h) / (0.5e0_rt * h + bcl);
            bfm = bfv - fac;

        }

    }
    else if (bct == AMREX_LO_NEUMANN) {

        bfm = -fac;
        stencil_weight_second = 0.e0_rt;

    }
    else {

        amrex::Error("fillLevelMat: unsupported boundary type");

    }

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k);
                mat(i,j,k)[1] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[2] += stencil_weight_second * b(i,j,k);
                }

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i+1,j,k);
                mat(i,j,k)[2] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[1] += stencil_weight_second * b(i+1,j,k);
                }

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k);
                mat(i,j,k)[3] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[4] += stencil_weight_second * b(i,j,k);
                }

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j+1,k);
                mat(i,j,k)[4] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[3] += stencil_weight_second * b(i,j+1,k);
                }

            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k);
                mat(i,j,k)[5] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[6] += stencil_weight_second * b(i,j,k);
                }

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k+1);
                mat(i,j,k)[6] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[5] += stencil_weight_second * b(i,j,k+1);
                }

            }

        }
    });

    Gpu::synchronize();
}

void
MultiLevelEllipticSolver::fillLevelMat3 (const Box& bx,
                        int side_is_low, int idir,
                        Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                        int cdir, int bctype,
                        Array4<int const> const& tf,
                        int bho, Real bcl,
                        Array4<int const> const& mask,
                        Array4<Real const> const& b,
                        Real beta, const Real* dx,
                        const GeometryData& geomdata, Real c,
                        Array4<Real const> const& spa)
{
    BL_PROFILE("MultiLevelEllipticSolver::fillLevelMat3");

    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h;

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    const Real fac = beta / (h * h);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real r;
        face_geometry_factors(i, j, k, bx.loVect()[0], bx.hiVect()[0], geomdata, idir, side_is_low, r);

        Real bfm, bfv;
        Real stencil_weight_second, h2, th2;
        int bct;

        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i-1,j,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k);
                    }
                    else  {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k);
                mat(i,j,k)[1] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[2] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i+1,j,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i+1,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i+1,j,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i+1,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i+1,j,k);
                mat(i,j,k)[2] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[1] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j-1,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k);
                mat(i,j,k)[3] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[4] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j+1,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j+1,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j+1,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j+1,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j+1,k);
                mat(i,j,k)[4] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[3] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k-1);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k);
                mat(i,j,k)[5] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[6] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k+1);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k+1);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k+1);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k+1);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k+1);
                mat(i,j,k)[6] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[5] += stencil_weight_second;
                }

            }

        }
    });

    Gpu::synchronize();
}

void MultiLevelEllipticSolver::fillBoundaryCoeffs (const Box& bx,
                           Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                           Array4<Real const> const& b,
                           Real beta, const Real* dx, int n)
{
    BL_PROFILE("MultiLevelEllipticSolver::fillBoundaryCoeffs");

    if (n == 0) {

        const Real fac = beta / (dx[0] * dx[0]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] += fac * (b(i,j,k) + b(i+1,j,k));
            mat(i,j,k)[1] = -fac * b(i,j,k);
            mat(i,j,k)[2] = -fac * b(i+1,j,k);
        });

    }
    else if (n == 1) {

        const Real fac = beta / (dx[1] * dx[1]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] += fac * (b(i,j,k) + b(i,j+1,k));
            mat(i,j,k)[3] = -fac * b(i,j,k);
            mat(i,j,k)[4] = -fac * b(i,j+1,k);
        });

    }
    else {

        const Real fac = beta / (dx[2] * dx[2]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] += fac * (b(i,j,k) + b(i,j,k+1));
            mat(i,j,k)[5] = -fac * b(i,j,k);
            mat(i,j,k)[6] = -fac * b(i,j,k+1);
        });

    }

    Gpu::synchronize();
}

void MultiLevelEllipticSolver::commitMatrix()
{
  BL_PROFILE("MultiLevelEllipticSolver::commitMatrix");

  HYPRE_SStructMatrixAssemble(A);
}

void MultiLevelEllipticSolver::fillLevelRhs(int level,
                                      MultiFab& rhs,
                                      BoundaryMode bc_variant)
{
  BL_PROFILE("MultiLevelEllipticSolver::fillLevelRhs");

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
      f->copy<RunOn::Device>(rhs[mfi]);
    }
    Elixir f_hold = scratch_fab.elixir();

    Real* vec = f->dataPtr();

    if (bc_variant) {
      const Box& domain = bd[level]->getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        int idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
        const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
        const FArrayBox       &fs  = bd[level]->bndryValues(oitr())[mfi];
        const Mask      &msk = bd[level]->bndryMasks(oitr(), i);
        const Box &bbox = (*bcoefs[level])[idim][mfi].box();
        const Box &msb  = msk.box();
        if (reg[oitr()] == domain[oitr()] || level == crse_level) {

          if (reg[oitr()] == domain[oitr()]) {
            Array4<const int> tfp{};
            int bctype = bct;
            if (bd[level]->isMixedFace(oitr())) {
              const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
              tfp = tf.array();
              bctype = -1;
            }
            AlphaBetaSolver::fillBcVec3(reg,
                              oitr().isLow(), idim,
                              f->array(),
                              cdir, bctype,
                              tfp,
                              bho, bcl,
                              fs.array(bc_component),
                              msk.array(),
                              (*bcoefs[level])[idim][mfi].array(),
                              beta, geom[level].data());
          }
          else {
              AlphaBetaSolver::fillBcVec(reg, f->array(),
                               cdir, bct, bho, bcl,
                               fs.array(bc_component), msk.array(),
                               (*bcoefs[level])[idim][mfi].array(),
                               beta, geom[level].CellSize());
          }
        }
      }
    }

    writeVectorBox(b, part, reg, subgrids[level][i], vec);
  }
}

void MultiLevelEllipticSolver::fillLevelVectors(int level,
                                      MultiFab& dest,
                                      int icomp,
                                      MultiFab& rhs,
                                      BoundaryMode bc_variant)
{
  BL_PROFILE("MultiLevelEllipticSolver::fillLevelVectors");

  int part = level - crse_level;

  Real *vec;
  FArrayBox scratch_fab;
  for (MFIter mfi(dest); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    FArrayBox *f;
    int fcomp;
    if (dest.nGrow() == 0) {
      f = &dest[mfi];
      fcomp = icomp;
    }
    else {
      f = &scratch_fab;
      f->resize(reg);
      f->copy<RunOn::Device>(dest[mfi], icomp, 0, 1);
      fcomp = 0;
    }
    Elixir f_hold = scratch_fab.elixir();

    vec = f->dataPtr(fcomp);

    writeVectorBox(x, part, reg, subgrids[level][i], vec);

    f->copy<RunOn::Device>(rhs[mfi], 0, fcomp, 1);

    if (bc_variant) {
      const Box& domain = bd[level]->getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        int idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
        const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
        const FArrayBox       &fs  = bd[level]->bndryValues(oitr())[mfi];
        const Mask      &msk = bd[level]->bndryMasks(oitr(), i);
        const Box &bbox = (*bcoefs[level])[idim][mfi].box();

        if (reg[oitr()] == domain[oitr()] || level == crse_level) {

          if (reg[oitr()] == domain[oitr()]) {
            Array4<const int> tfp{};
            int bctype = bct;
            if (bd[level]->isMixedFace(oitr())) {
              const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
              tfp = tf.array();
              bctype = -1;
            }
            AlphaBetaSolver::fillBcVec3(reg,
                              oitr().isLow(), idim,
                              f->array(fcomp),
                              cdir, bctype,
                              tfp,
                              bho, bcl,
                              fs.array(bc_component),
                              msk.array(),
                              (*bcoefs[level])[idim][mfi].array(),
                              beta, geom[level].data());
          }
          else {
              AlphaBetaSolver::fillBcVec(reg, f->array(fcomp),
                               cdir, bct, bho, bcl,
                               fs.array(bc_component), msk.array(),
                               (*bcoefs[level])[idim][mfi].array(),
                               beta, geom[level].CellSize());
          }
        }
      }
    }

    writeVectorBox(b, part, reg, subgrids[level][i], vec);
  }
}

void MultiLevelEllipticSolver::commitVectors()
{
  BL_PROFILE("MultiLevelEllipticSolver::commitVectors");

  HYPRE_SStructVectorAssemble(b);
  HYPRE_SStructVectorAssemble(x);
}

void MultiLevelEllipticSolver::loadLevelSolution(int level,
                                      MultiFab& dest,
                                      int icomp)
{
  BL_PROFILE("MultiLevelEllipticSolver::loadLevelSolution");

  int part = level - crse_level;

  FArrayBox scratch_fab;
  for (MFIter mfi(dest); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    FArrayBox *f;
    int fcomp;
    if (dest.nGrow() == 0) {
      f = &dest[mfi];
      fcomp = icomp;
    }
    else {
      f = &scratch_fab;
      f->resize(reg);
      f->copy<RunOn::Device>(dest[mfi], icomp, 0, 1);
      fcomp = 0;
    }
    Elixir f_hold = scratch_fab.elixir();

    Real* vec = f->dataPtr(fcomp);

    writeVectorBox(x, part, reg, subgrids[level][i], vec);
  }
}
