#ifndef HYDROSIM_RAD_BOUNDARYCOND_H
#define HYDROSIM_RAD_BOUNDARYCOND_H

#include <AMReX_BLassert.H>

//@Man:
/*@Memo:
        Maintain an identifier for boundary condition types.
*/
/*@Doc:
        This is a placeholder for more extensive boundary condition
        implementations, which might include stencils, etc.  Presently,
        boundary conditions are specified via an integer identifier.
        This class maintains that integer.
*/
class RadBoundaryCondition
{
public:
//@ManMemo: administrative functions
//@ManDoc: constructor, initializes boundary condition type to default
    RadBoundaryCondition() : bctype(-1){}
//@ManDoc: constructor, initializes boundary condition type to specified value
    RadBoundaryCondition(int _bctype) : bctype(_bctype){}

//@ManMemo: access functions
//@ManDoc: return index of boundary condition type
    operator int () const
    {
        BL_ASSERT(bctype != -1);
        return bctype;
    }
//@ManDoc: set index of boundary condition type
    RadBoundaryCondition& operator = (const int _bctype)
    {
        bctype = _bctype;
        return *this;
    }
private:
//@ManDoc: index of boundary condition type
    int bctype;
};

#endif
