#ifndef HYDROSIM_RAD_BOUNDARYCOND_H
#define HYDROSIM_RAD_BOUNDARYCOND_H

#include <AMReX_BLassert.H>

class RadBoundaryCondition
{
public:
    RadBoundaryCondition() : bctype(-1){}
    RadBoundaryCondition(int _bctype) : bctype(_bctype){}

    operator int () const
    {
        BL_ASSERT(bctype != -1);
        return bctype;
    }
    RadBoundaryCondition& operator = (const int _bctype)
    {
        bctype = _bctype;
        return *this;
    }
private:
    int bctype;
};

#endif
