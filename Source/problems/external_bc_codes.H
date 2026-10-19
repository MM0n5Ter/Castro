#ifndef EXTERNAL_BC_CODES_H
#define EXTERNAL_BC_CODES_H

constexpr int EXTERNAL_UNDEFINED = -1;
constexpr int EXTERNAL_HYDROSTATIC = 1;

namespace hse
{
    constexpr int MAX_ITER = 250;
    constexpr Real TOL = 1.e-8_rt;
}

#endif
