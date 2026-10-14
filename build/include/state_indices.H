#ifndef STATE_INDICES_H
#define STATE_INDICES_H
#include <network_properties.H>

  constexpr int NumAdv = 0;
  constexpr int NQAUX = 4 + NGROUPS;
  constexpr int NUM_STATE = 7 + NumAdv + NumSpec + NumAux;
  constexpr int NSRC = 7;
  constexpr int NIFC = 4 + NGROUPS + NGROUPS;
  constexpr int NQ = 9 + NGROUPS + NumAdv + NumSpec + NumAux;
  constexpr int NQSRC = 7;
  constexpr int NQTHERM = 9 + NGROUPS;
  constexpr int npassive = NumSpec + NumAux + NumAdv;

   // auxiliary
  constexpr int QGAM1 = 0;
  constexpr int QC = 1;
  constexpr int QGAM1GAS = 2;
  constexpr int QCSNDGAS = 3;
  constexpr int QFLXLIM = 4;

   // conserved
  constexpr int URHO = 0;
  constexpr int UMX = 1;
  constexpr int UMY = 2;
  constexpr int UMZ = 3;
  constexpr int UEDEN = 4;
  constexpr int UEINT = 5;
  constexpr int UTEMP = 6;
  constexpr int UADV = 7;
  constexpr int USPEC = 7 + NumAdv;
  constexpr int UAUXV = 7 + NumAdv + NumSpec;

   // godunov
  constexpr int IFCU = 0;
  constexpr int IFCV = 1;
  constexpr int IFCW = 2;
  constexpr int IFCPRES = 3;
  constexpr int IFCFLXLIM = 4;
  constexpr int IFCERAD = 4 + NGROUPS;

   // primitive
  constexpr int QRHO = 0;
  constexpr int QU = 1;
  constexpr int QV = 2;
  constexpr int QW = 3;
  constexpr int QPRES = 4;
  constexpr int QRHOEI = 5;
  constexpr int QTEMP = 6;
  constexpr int QPTOTAL = 7;
  constexpr int QRHOEITOT = 8;
  constexpr int QERAD = 9;
  constexpr int QADV = 9 + NGROUPS;
  constexpr int QSPEC = 9 + NGROUPS + NumAdv;
  constexpr int QAUXV = 9 + NGROUPS + NumAdv + NumSpec;

#endif
