
void hs_dergradpoverp
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_dergradpoverp1
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_dergradpx
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_dergradpy
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_dergradrhooverrho
  (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
   const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
   amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);
