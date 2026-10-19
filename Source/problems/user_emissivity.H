#ifndef user_emissivity_H
#define user_emissivity_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_emissivity (int i, int j, int k, int g,
                         const GpuArray<Real, NGROUPS>& nu,
                         const GpuArray<Real, NGROUPS+1>& xnu,
                         Real T, Real kg, Real dOpacT,
                         Real& jg, Real& dEmitT) {}

#endif
