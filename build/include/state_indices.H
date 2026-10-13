#ifndef STATE_INDICES_H
#define STATE_INDICES_H
#include <network_properties.H>

  constexpr int NumAdv = 0;
  constexpr int NQAUX = 2;
  constexpr int NUM_STATE = 7 + NumAdv + NumSpec + NumAux;
  constexpr int NSRC = 7;
  constexpr int NGDNV = 4;
  constexpr int NQ = 11 + NumAdv + NumSpec + NumAux;
  constexpr int NQSRC = 10;
  constexpr int NQTHERM = 11;
  constexpr int npassive = NumSpec + NumAux + NumAdv;

   // auxiliary
  constexpr int QGAMC = 0;
  constexpr int QC = 1;

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

   // primitive
  constexpr int QRHO = 0;
  constexpr int QU = 1;
  constexpr int QV = 2;
  constexpr int QW = 3;
  constexpr int QPRES = 4;
  constexpr int QREINT = 5;
  constexpr int QMAGX = 6;
  constexpr int QMAGY = 7;
  constexpr int QMAGZ = 8;
  constexpr int QTEMP = 9;
  constexpr int QPTOT = 10;
  constexpr int QFA = 11;
  constexpr int QFS = 11 + NumAdv;
  constexpr int QFX = 11 + NumAdv + NumSpec;

#endif
