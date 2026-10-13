#ifndef STATE_INDICES_H
#define STATE_INDICES_H
#include <network_properties.H>

  constexpr int NumAdv = 0;
  constexpr int NQAUX = 4 + NGROUPS;
  constexpr int NUM_STATE = 7 + NumAdv + NumSpec + NumAux;
  constexpr int NSRC = 7;
  constexpr int NGDNV = 4 + NGROUPS + NGROUPS;
  constexpr int NQ = 9 + NGROUPS + NumAdv + NumSpec + NumAux;
  constexpr int NQSRC = 7;
  constexpr int NQTHERM = 9 + NGROUPS;
  constexpr int npassive = NumSpec + NumAux + NumAdv;

   // auxiliary
  constexpr int QGAMC = 0;
  constexpr int QC = 1;
  constexpr int QGAMCG = 2;
  constexpr int QCG = 3;
  constexpr int QLAMS = 4;

   // conserved
  constexpr int URHO = 0;
  constexpr int UMX = 1;
  constexpr int UMY = 2;
  constexpr int UMZ = 3;
  constexpr int UEDEN = 4;
  constexpr int UEINT = 5;
  constexpr int UTEMP = 6;
  constexpr int UFA = 7;
  constexpr int UFS = 7 + NumAdv;
  constexpr int UFX = 7 + NumAdv + NumSpec;

   // godunov
  constexpr int GDU = 0;
  constexpr int GDV = 1;
  constexpr int GDW = 2;
  constexpr int GDPRES = 3;
  constexpr int GDLAMS = 4;
  constexpr int GDERADS = 4 + NGROUPS;

   // primitive
  constexpr int QRHO = 0;
  constexpr int QU = 1;
  constexpr int QV = 2;
  constexpr int QW = 3;
  constexpr int QPRES = 4;
  constexpr int QREINT = 5;
  constexpr int QTEMP = 6;
  constexpr int QPTOT = 7;
  constexpr int QREITOT = 8;
  constexpr int QRAD = 9;
  constexpr int QFA = 9 + NGROUPS;
  constexpr int QFS = 9 + NGROUPS + NumAdv;
  constexpr int QFX = 9 + NGROUPS + NumAdv + NumSpec;

#endif
