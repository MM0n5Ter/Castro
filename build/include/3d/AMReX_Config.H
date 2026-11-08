#ifndef AMREX_HAVE_NO_CONFIG_H
#define AMREX_HAVE_NO_CONFIG_H
#ifndef BL_LAZY
#define BL_LAZY 1
#endif
#ifndef AMREX_LAZY
#define AMREX_LAZY 1
#endif
#ifndef BL_NO_FORT
#define BL_NO_FORT 1
#endif
#ifndef AMREX_GPU_MAX_THREADS
#define AMREX_GPU_MAX_THREADS 0
#endif
#ifndef BL_SPACEDIM
#define BL_SPACEDIM 3
#endif
#ifndef AMREX_SPACEDIM
#define AMREX_SPACEDIM 3
#endif
#ifndef BL_FORT_USE_UNDERSCORE
#define BL_FORT_USE_UNDERSCORE 1
#endif
#ifndef AMREX_FORT_USE_UNDERSCORE
#define AMREX_FORT_USE_UNDERSCORE 1
#endif
#ifndef BL_Linux
#define BL_Linux 1
#endif
#ifndef AMREX_Linux
#define AMREX_Linux 1
#endif
#ifndef AMREX_DIMENSION_AGNOSTIC
#define AMREX_DIMENSION_AGNOSTIC 1
#endif
#ifndef NDEBUG
#define NDEBUG 1
#endif
#ifndef AMREX_NO_PROBINIT
#define AMREX_NO_PROBINIT 1
#endif
#ifndef SPONGE
#define SPONGE 1
#endif
#ifndef REACTIONS
#define REACTIONS 1
#endif
#ifndef TRUE_SDC
#define TRUE_SDC 1
#endif
#ifndef SDC
#define SDC 1
#endif
#ifndef NET_LOOP_UNROLL_LEN
#define NET_LOOP_UNROLL_LEN 4
#endif
#ifndef SCREEN_METHOD
#define SCREEN_METHOD SCREEN_METHOD_screen5
#endif
#ifndef NEUTRINO_METHOD
#define NEUTRINO_METHOD NEUTRINO_METHOD_sneut5
#endif
#ifndef AUTODIFF_DEVICE_FUNC
#define AUTODIFF_DEVICE_FUNC AMREX_GPU_HOST_DEVICE
#endif
#ifndef ALLOW_JACOBIAN_CACHING
#define ALLOW_JACOBIAN_CACHING 1
#endif
#ifndef RATES
#define RATES 1
#endif
#ifndef SCREENING
#define SCREENING 1
#endif
#ifndef NAUX_NET
#define NAUX_NET 0
#endif
#ifdef __cplusplus
#endif
#if defined(AMREX_USE_OMP) && !defined(_OPENMP) && !defined(AMREX_USE_HIP)
#error libamrex was built with OpenMP
#endif
#endif
