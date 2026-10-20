#include <limits.h>
#include <math.h>
#include <AMReX_LO_BCTYPES.H>
#include <RadInterpBoundaryData.H>

#include <AMReX_Utility.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

#if (AMREX_SPACEDIM == 2)
#define DERIV_TERMS 2
#endif

#if (AMREX_SPACEDIM == 3)
#define DERIV_TERMS 5
#endif

#define DECLARE_EXTENTS(fab,fab_values,fab_low_corner,fab_high_corner)   \
const int* fab_low_corner = (fab).loVect();           \
const int* fab_high_corner = (fab).hiVect();           \
REAL* fab_values = (fab).dataPtr();
#define DECLARE_CONST_EXTENTS(fab,fab_values,fab_low_corner,fab_high_corner)  \
const int* fab_low_corner = (fab).loVect();           \
const int* fab_high_corner = (fab).hiVect();           \
const REAL* fab_values = (fab).dataPtr();

RadBoundaryData::RadBoundaryData(const BoxArray& _grids, const DistributionMapping& _dmap,
                           int _ncomp, const GeometryStandIn& _geom)
    : BndryRegister(), geom(_geom)
{
    define(_grids,_dmap,_ncomp,_geom);
}

std::ostream& operator << (std::ostream& os, const RadBoundaryData &mgb)
{
    const BoxArray& grds = mgb.boxes();
    int ngrds = grds.size();
    os << "[RadBoundaryData with " << ngrds << " grids:\n";
    for (int grd = 0; grd < ngrds; grd++){
        for (OrientationIter face; face; ++face) {
            Orientation f = face();
            os << "::: face " << f << " of grid " << grds[grd] << "\n";
            os << "BC = " << mgb.bcond[f][grd]
               << " LOC = " << mgb.bcloc[f][grd] << "\n";
            os << *mgb.masks[f][grd];
            os << mgb.bndry[f][grd];
        }
        os << "--------------------------------------------------" << std::endl;
    }
    return os;
}

void
RadBoundaryData::define(const BoxArray& _grids, const DistributionMapping& _dmap,
                     int _ncomp, const GeometryStandIn& _geom)
{
    geom = _geom;
    BndryRegister::setBoxes(_grids);
    int len = grids.size();
    BL_ASSERT( len > 0 );

    for (OrientationIter fi; fi; ++fi) {
        Orientation face = fi();
        int face_axis = face.coordDir();
        masks[face].resize(len);
        bcloc[face].resize(len);
        bcond[face].resize(len);

        BndryRegister::define(face,IndexType::TheCellType(),0,1,0,_ncomp,_dmap);

        for (FabSetIter bi(bndry[face]); bi.isValid(); ++bi) {
            int k = bi.index();
            Box face_box = amrex::adjCell(grids[k],face,1);

            for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                if (dir == face_axis) continue;
                face_box.grow(dir,1);
            }
            masks[face][k].reset(new Mask(face_box));
            Mask *m = masks[face][k].get();
            m->setVal<RunOn::Host>(outside_domain,0);
            Box dbox(geom.Domain());
            dbox &= face_box;
            m->setVal<RunOn::Host>(not_covered,dbox,0);
            if( geom.isAnyPeriodic() ){
              Box dombox(geom.Domain());
              Vector<IntVect> pshifts(27);
              geom.periodicShift( dombox, face_box, pshifts );
              for( int iiv=0; iiv<pshifts.size(); iiv++){
                IntVect iv = pshifts[iiv];
                m->shift(iv);
                Box target(dombox);
                target &= m->box();
                if (target.ok()) m->setVal<RunOn::Host>(not_covered,target,0);
                m->shift(-iv);
              }
            }
            for (int g = 0; g < len; g++) {
                Box ovlp(grids[g]);
                ovlp &= face_box;
                if (ovlp.ok()) m->setVal<RunOn::Host>(covered,ovlp,0);
            }
            if( geom.isAnyPeriodic() &&
                !geom.Domain().contains(face_box) ){
              Vector<IntVect> pshifts(27);
              geom.periodicShift( geom.Domain(), face_box, pshifts);
              for( int iiv=0; iiv<pshifts.size(); iiv++ ){
                IntVect iv = pshifts[iiv];
                m->shift(iv);
                for( int g=0; g<len; g++){
                  Box ovlp(grids[g]);
                  ovlp &= m->box();
                  if( ovlp.ok() ) m->setVal<RunOn::Host>(covered,ovlp,0);
                }
                m->shift(-iv);
              }
            }
        }
    }
}

RadInterpBoundaryData::RadInterpBoundaryData(const BoxArray& _grids, const DistributionMapping& _dmap,
                                       int _ncomp, const Geometry& geom)
    : RadBoundaryData(_grids,_dmap,_ncomp,geom)
{
}

void
RadInterpBoundaryData::setBndryValues(Real bv)
{
  for (OrientationIter fi; fi; ++fi) {
      bndry[fi()].setVal(bv);
  }
}

void
RadInterpBoundaryData::setBndryValues(const MultiFab& mf, int mf_start,
                                int bnd_start, int num_comp,
                                const BCRec& bc )
{
    BL_ASSERT( grids.size() );
    BL_ASSERT( grids == mf.boxArray() );

    IntVect ref_ratio = IntVect::TheUnitVector();
    setBoundaryConditions(bc, geom, ref_ratio);

    for(MFIter mfi(mf); mfi.isValid(); ++mfi) {
        BL_ASSERT(grids[mfi.index()] == mfi.validbox());
        int grd = mfi.index();
        const Box& bx = grids[grd];
        for (OrientationIter fi; fi; ++fi) {
            Orientation face(fi());
            if (bx[face] == geom.Domain()[face]) {
                FArrayBox& bnd_fab = bndry[face][mfi];
                bnd_fab.copy<RunOn::Host>(mf[mfi],mf_start,bnd_start,num_comp);
            }
        }
    }

    for (OrientationIter fi; fi; ++fi) {
        bndry[fi()].copyFrom(mf,0,mf_start,bnd_start,num_comp);
    }
}

void
RadInterpBoundaryData::setBndryValues(BndryRegister& crse, int c_start,
                                const MultiFab& fine, int f_start,
                                int bnd_start, int num_comp, IntVect& ratio,
                                const BCRec& bc)
{
    BL_ASSERT( grids.size() );
    BL_ASSERT( grids == fine.boxArray() );

    setBoundaryConditions(bc, geom, ratio);

    const Box& fine_domain = geom.Domain();
    Real *derives = 0;
    int  scratch_len = 0;
    for(MFIter mfi(fine); mfi.isValid(); ++mfi) {
        BL_ASSERT(grids[mfi.index()] == mfi.validbox());
        int grd = mfi.index();
        const Box& fine_bx = grids[grd];
        Box crse_bx = amrex::coarsen(fine_bx,ratio);
        auto coarse_low_corner = crse_bx.loVect3d();
        auto coarse_high_corner = crse_bx.hiVect3d();
        int coarse_side_len = crse_bx.longside() + 2;
        if (std::pow(coarse_side_len,(float)AMREX_SPACEDIM-1) > scratch_len) {
            delete [] derives;
#if (AMREX_SPACEDIM == 1)
            derives = new Real[1];
#else
            scratch_len = coarse_side_len;
#  if (AMREX_SPACEDIM > 2)
            scratch_len *= coarse_side_len;
#  endif
            derives = new Real[scratch_len*DERIV_TERMS];
#endif
        }
        auto lo = fine_bx.loVect3d();
        auto hi = fine_bx.hiVect3d();
        const FArrayBox& fine_patch = fine[mfi];

        for (OrientationIter fi; fi; ++fi) {
            Orientation face(fi());
            int dir = face.coordDir();
            if (fine_bx[face] != fine_domain[face] ||
                geom.isPeriodic(dir)) {
                const Mask& mask = *masks[face][grd];
                Array4<int const> const mask_arr = mask.array();

                const FArrayBox& crse_fab = crse[face][mfi];
                Array4<Real const> const crse_arr = crse_fab.array(c_start);

                int clo[3], chi[3], flo[3], fhi[3];

                for (int d = 0; d < 3; ++d) {
                    clo[d] = coarse_low_corner[d];
                    chi[d] = coarse_high_corner[d];
                    flo[d] = lo[d];
                    fhi[d] = hi[d];


                    if (d == dir) {
                        if (face.isLow()) {
                            clo[d] -= 1;
                            flo[d] -= 1;
                            chi[d] = clo[d];
                            fhi[d] = flo[d];
                        }
                        else {
                            chi[d] += 1;
                            fhi[d] += 1;
                            clo[d] = chi[d];
                            flo[d] = fhi[d];
                        }
                    }
                }

                int ratio_x = 1;
                int ratio_y = 1;
                int ratio_z = 1;

                if (dir != 0) {
                    ratio_x = ratio[0];
                }

#if AMREX_SPACEDIM >= 2
                if (dir != 1) {
                    ratio_y = ratio[1];
                }
#endif

#if AMREX_SPACEDIM == 3
                if (dir != 2) {
                    ratio_z = ratio[2];
                }
#endif

                FArrayBox& bnd_fab = bndry[face][mfi];
                Array4<Real> const bdry = bnd_fab.array(bnd_start);

                int is_not_covered = RadBoundaryData::not_covered;

                for (int n = 0; n < num_comp; ++n) {


                    for (int kc = clo[2]; kc <= chi[2]; ++kc) {
                        int k = (dir == 2) ? flo[2] : ratio_z * kc;

                        for (int jc = clo[1]; jc <= chi[1]; ++jc) {
                            int j = (dir == 1) ? flo[1] : ratio_y * jc;

                            for (int ic = clo[0]; ic <= chi[0]; ++ic) {
                                int i = (dir == 0) ? flo[0] : ratio_x * ic;

                                Real slope_x = 0.0_rt;
                                Real slope_y = 0.0_rt;
                                Real slope_z = 0.0_rt;

                                Real curvature_x = 0.0_rt;
                                Real curvature_y = 0.0_rt;
                                Real curvature_z = 0.0_rt;

                                Real cross_slope_xy = 0.0_rt;
                                Real cross_slope_xz = 0.0_rt;
                                Real cross_slope_yz = 0.0_rt;

                                if (dir != 0) {
                                    slope_x = 0.5_rt * (crse_arr(ic+1,jc,kc,n) - crse_arr(ic-1,jc,kc,n));
                                    curvature_x = 0.5_rt * (crse_arr(ic+1,jc,kc,n) - 2.0_rt * crse_arr(ic,jc,kc,n) + crse_arr(ic-1,jc,kc,n));
                                }

#if AMREX_SPACEDIM >= 2
                                if (dir != 1) {
                                    slope_y = 0.5_rt * (crse_arr(ic,jc+1,kc,n) - crse_arr(ic,jc-1,kc,n));
                                    curvature_y = 0.5_rt * (crse_arr(ic,jc+1,kc,n) - 2.0_rt * crse_arr(ic,jc,kc,n) + crse_arr(ic,jc-1,kc,n));
                                }
#endif

#if AMREX_SPACEDIM == 3
                                if (dir != 2) {
                                    slope_z = 0.5_rt * (crse_arr(ic,jc,kc+1,n) - crse_arr(ic,jc,kc-1,n));
                                    curvature_z = 0.5_rt * (crse_arr(ic,jc,kc+1,n) - 2.0_rt * crse_arr(ic,jc,kc,n) + crse_arr(ic,jc,kc-1,n));
                                }
#endif

#if AMREX_SPACEDIM == 3
                                if (dir == 0) {
                                    cross_slope_yz = 0.25_rt * (crse_arr(ic,jc+1,kc+1,n) - crse_arr(ic,jc-1,kc+1,n) +
                                                       crse_arr(ic,jc-1,kc-1,n) - crse_arr(ic,jc+1,kc-1,n));
                                }

                                if (dir == 1) {
                                    cross_slope_xz = 0.25_rt * (crse_arr(ic+1,jc,kc+1,n) - crse_arr(ic-1,jc,kc+1,n) +
                                                       crse_arr(ic-1,jc,kc-1,n) - crse_arr(ic+1,jc,kc-1,n));
                                }

                                if (dir == 2) {
                                    cross_slope_xy = 0.25_rt * (crse_arr(ic+1,jc+1,kc,n) - crse_arr(ic-1,jc+1,kc,n) +
                                                       crse_arr(ic-1,jc-1,kc,n) - crse_arr(ic+1,jc-1,kc,n));
                                }
#endif

                                if (dir != 0) {
                                    if (mask_arr(i-1,j,k) != is_not_covered) {
                                        slope_x = crse_arr(ic+1,jc,kc,n) - crse_arr(ic,jc,kc,n);
                                        curvature_x = 0.0_rt;
                                    }
                                    if (mask_arr(i+ratio_x,j,k) != is_not_covered) {
                                        slope_x = crse_arr(ic,jc,kc,n) - crse_arr(ic-1,jc,kc,n);
                                        curvature_x = 0.0_rt;
                                    }
                                    if (mask_arr(i-1,j,k) != is_not_covered && mask_arr(i+ratio_x,j,k) != is_not_covered) {
                                        slope_x = 0.0_rt;
                                    }
                                }

#if AMREX_SPACEDIM >= 2
                                if (dir != 1) {
                                    if (mask_arr(i,j-1,k) != is_not_covered) {
                                        slope_y = crse_arr(ic,jc+1,kc,n) - crse_arr(ic,jc,kc,n);
                                        curvature_y = 0.0_rt;
                                    }
                                    if (mask_arr(i,j+ratio_y,k) != is_not_covered) {
                                        slope_y = crse_arr(ic,jc,kc,n) - crse_arr(ic,jc-1,kc,n);
                                        curvature_y = 0.0_rt;
                                    }
                                    if (mask_arr(i,j-1,k) != is_not_covered && mask_arr(i,j+ratio_y,k) != is_not_covered) {
                                        slope_y = 0.0_rt;
                                    }
                                }
#endif

#if AMREX_SPACEDIM == 3
                                if (dir != 2) {
                                    if (mask_arr(i,j,k-1) != is_not_covered) {
                                        slope_z = crse_arr(ic,jc,kc+1,n) - crse_arr(ic,jc,kc,n);
                                        curvature_z = 0.0_rt;
                                    }
                                    if (mask_arr(i,j,k+ratio_z) != is_not_covered) {
                                        slope_z = crse_arr(ic,jc,kc,n) - crse_arr(ic,jc,kc-1,n);
                                        curvature_z = 0.0_rt;
                                    }
                                    if (mask_arr(i,j,k-1) != is_not_covered && mask_arr(i,j,k+ratio_z) != is_not_covered) {
                                        slope_z = 0.0_rt;
                                    }
                                }
#endif

#if AMREX_SPACEDIM == 3
                                if (dir == 0) {
                                    if ((mask_arr(i,j+ratio_y,k+ratio_z) != is_not_covered) ||
                                        (mask_arr(i,j-1     ,k+ratio_z) != is_not_covered) ||
                                        (mask_arr(i,j+ratio_y,k-1     ) != is_not_covered) ||
                                        (mask_arr(i,j-1     ,k-1     ) != is_not_covered)) {
                                        cross_slope_yz = 0.0_rt;
                                    }
                                }

                                if (dir == 1) {
                                    if ((mask_arr(i+ratio_x,j,k+ratio_z) != is_not_covered) ||
                                        (mask_arr(i-1     ,j,k+ratio_z) != is_not_covered) ||
                                        (mask_arr(i+ratio_x,j,k-1     ) != is_not_covered) ||
                                        (mask_arr(i-1     ,j,k-1     ) != is_not_covered)) {
                                        cross_slope_xz = 0.0_rt;
                                    }
                                }

                                if (dir == 2) {
                                    if ((mask_arr(i+ratio_x,j+ratio_y,k) != is_not_covered) ||
                                        (mask_arr(i-1     ,j+ratio_y,k) != is_not_covered) ||
                                        (mask_arr(i+ratio_x,j-1     ,k) != is_not_covered) ||
                                        (mask_arr(i-1     ,j-1     ,k) != is_not_covered)) {
                                        cross_slope_xy = 0.0_rt;
                                    }
                                }
#endif

                                for (int koff = 0; koff < ratio_z; ++koff) {
                                    Real zz = (koff - 0.5_rt * ratio_z + 0.5_rt) / ratio_z;
                                    int kk = (dir == 2) ? flo[2] : ratio_z * kc + koff;

                                    for (int joff = 0; joff < ratio_y; ++joff) {
                                        Real yy = (joff - 0.5_rt * ratio_y + 0.5_rt) / ratio_y;
                                        int jj = (dir == 1) ? flo[1] : ratio_y * jc + joff;

                                        for (int ioff = 0; ioff < ratio_x; ++ioff) {
                                            Real xx = (ioff - 0.5_rt * ratio_x + 0.5_rt) / ratio_x;
                                            int ii = (dir == 0) ? flo[0] : ratio_x * ic + ioff;

                                            bdry(ii,jj,kk,n) = crse_arr(ic,jc,kc,n) +
                                                               xx * slope_x + xx * xx * curvature_x +
                                                               yy * slope_y + yy * yy * curvature_y +
                                                               zz * slope_z + zz * zz * curvature_z +
                                                               xx * yy * cross_slope_xy + xx * zz * cross_slope_xz + yy * zz * cross_slope_yz;
                                        }
                                    }
                                }
                            }
                        }
                    }

                }
            } else {
                FArrayBox& bnd_fab = bndry[face][mfi];
                bnd_fab.copy<RunOn::Host>(fine_patch,f_start,bnd_start,num_comp);
            }
        }
    }
    delete[] derives;

    for (OrientationIter face; face; ++face) {
        bndry[face()].copyFrom(fine,0,f_start,bnd_start,num_comp);
    }
}

void RadInterpBoundaryData::setBoundaryConditions(const BCRec& phys_bc,
                                    const Geometry& geom, int ratio)
{

    IntVect uniform_ratio = ratio * IntVect::TheUnitVector();
    setBoundaryConditions(phys_bc, geom, uniform_ratio);

}

