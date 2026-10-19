program bc

  use blackbody_module, only : PlanckBandIntegral

  implicit none

  integer :: ngroups, rad_group_idx
  real(rt)        , allocatable :: xnu(:)
  real(rt)         :: T

  integer :: n
  character(len=256) :: line
  integer :: ipos

  ! open the file
  open(unit=10,file="group_structure.dat")

  ! read in the number of groups
  read(10,fmt='(a256)') line
  ipos = index(line, "=") + 1
  read (line(ipos:),*) ngroups

  allocate (xnu(0:ngroups))

  ! skip lines
  do n=1,ngroups+3
     read(10,fmt='(a256)') line
  end do

  do rad_group_idx = 0, ngroups
     read(10,fmt='(a256)') line
     ipos = index(line, "=") + 1
     read (line(ipos:),*) xnu(rad_group_idx)
  enddo

  print *, 'T = ?'
  read(*,*) T

  print *, "T = ", T
  print *, ((PlanckBandIntegral(T, xnu(rad_group_idx), xnu(rad_group_idx+1))), rad_group_idx=0,ngroups-1)

end program bc
