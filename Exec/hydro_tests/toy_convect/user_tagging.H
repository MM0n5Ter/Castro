#ifndef user_tagging_H
#define user_tagging_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_tagging(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level, const GeometryData& geomdata)
{

    if (state(i,j,k,URHO) > problem::cutoff_density && state(i,j,k,UFS) > problem::H_min) {
        tag(i,j,k) = TagBox::SET;
    }
}

#endif
