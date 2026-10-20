#ifndef STATE_INDICES_H
#define STATE_INDICES_H
#include <network_properties.H>

  constexpr int NumAdvected = 0;
  constexpr int NQEXTRAQ = 2;
  constexpr int NUM_STATE = 7 + NumAdvected + NumSpec + NumAux;
  constexpr int NUM_SOURCE = 7;
  constexpr int NGDSTATE = 4;
  constexpr int NQ = 7 + NumAdvected + NumSpec + NumAux;
  constexpr int NQPRIMBASE = 7;
  constexpr int NQSOURCE = 7;
  constexpr int npassive = NumSpec + NumAux + NumAdvected;

   // auxiliary
  constexpr int QGAMMA1 = 0;
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
  constexpr int UFS = 7 + NumAdvected;
  constexpr int UFX = 7 + NumAdvected + NumSpec;

   // godunov
  constexpr int GDU = 0;
  constexpr int GDV = 1;
  constexpr int GDW = 2;
  constexpr int GDPRESSURE = 3;

   // primitive
  constexpr int QRHO = 0;
  constexpr int QU = 1;
  constexpr int QV = 2;
  constexpr int QW = 3;
  constexpr int QPRES = 4;
  constexpr int QEINTDENS = 5;
  constexpr int QTEMP = 6;
  constexpr int QFA = 7;
  constexpr int QFS = 7 + NumAdvected;
  constexpr int QFX = 7 + NumAdvected + NumSpec;

#endif
