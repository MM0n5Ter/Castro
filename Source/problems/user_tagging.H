#ifndef user_tagging_H
#define user_tagging_H


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_tagging(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level, const GeometryData& geomdata) {

    amrex::ignore_unused(i);
    amrex::ignore_unused(j);
    amrex::ignore_unused(k);
    amrex::ignore_unused(tag);
    amrex::ignore_unused(state);
    amrex::ignore_unused(level);
    amrex::ignore_unused(geomdata);

}

#endif
