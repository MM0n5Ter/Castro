#ifndef STATE_INDICES_H
#define STATE_INDICES_H
#include <network_properties.H>

  constexpr int NumAdv = 0;
  constexpr int NQAUX = 2;
  constexpr int NUM_STATE = 7 + NumAdv + NumSpec + NumAux;
  constexpr int NSRC = 7;
  constexpr int NIFC = 4;
  constexpr int NQ = 7 + NumAdv + NumSpec + NumAux;
  constexpr int NQSRC = 7;
  constexpr int NQTHERM = 7;
  constexpr int npassive = NumSpec + NumAux + NumAdv;

   // auxiliary
  constexpr int QGAM1 = 0;
  constexpr int QC = 1;

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

   // primitive
  constexpr int QRHO = 0;
  constexpr int QU = 1;
  constexpr int QV = 2;
  constexpr int QW = 3;
  constexpr int QPRES = 4;
  constexpr int QRHOEI = 5;
  constexpr int QTEMP = 6;
  constexpr int QADV = 7;
  constexpr int QSPEC = 7 + NumAdv;
  constexpr int QAUXV = 7 + NumAdv + NumSpec;

#endif
