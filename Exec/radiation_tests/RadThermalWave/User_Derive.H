
void dertexact
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int  ,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real  , const int*  , int  );

void derterror
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int  ,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real  , const int*  , int  );
