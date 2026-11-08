#ifndef STATE_INDICES_H
#define STATE_INDICES_H
#include <network_properties.H>

  constexpr int NumAdvected = 0;
  constexpr int NQEXTRAQ = 4 + NGROUPS;
  constexpr int NUM_STATE = 7 + NumAdvected + NumSpec + NumAux;
  constexpr int NUM_SOURCE = 7;
  constexpr int NGDSTATE = 4 + NGROUPS + NGROUPS;
  constexpr int NQ = 9 + NGROUPS + NumAdvected + NumSpec + NumAux;
  constexpr int NQPRIMBASE = 9 + NGROUPS;
  constexpr int NQSOURCE = 7;
  constexpr int npassive = NumSpec + NumAux + NumAdvected;

   // auxiliary
  constexpr int QGAMMA1 = 0;
  constexpr int QC = 1;
  constexpr int QGAMMA1G = 2;
  constexpr int QCG = 3;
  constexpr int QFLUXLIM = 4;

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
  constexpr int GDFLUXLIM = 4;
  constexpr int GDRADENG = 4 + NGROUPS;

   // primitive
  constexpr int QRHO = 0;
  constexpr int QU = 1;
  constexpr int QV = 2;
  constexpr int QW = 3;
  constexpr int QPRES = 4;
  constexpr int QEINTDENS = 5;
  constexpr int QTEMP = 6;
  constexpr int QPTOTAL = 7;
  constexpr int QETOTDENS = 8;
  constexpr int QRADENG = 9;
  constexpr int QFA = 9 + NGROUPS;
  constexpr int QFS = 9 + NGROUPS + NumAdvected;
  constexpr int QFX = 9 + NGROUPS + NumAdvected + NumSpec;

#endif
