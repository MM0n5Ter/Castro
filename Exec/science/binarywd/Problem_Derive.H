
void hs_derinertialmomentumx
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialmomentumy
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialmomentumz
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialangmomx
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialangmomy
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialangmomz
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialradmomx
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialradmomy
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derinertialradmomz
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derphieff
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derphieffpm_p
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derphieffpm_s
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derrhophiGrav
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derrhophiRot
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_derprimarymask
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

void hs_dersecondarymask
(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);
