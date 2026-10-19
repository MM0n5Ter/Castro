#ifndef ANGULAR_VELOCITY_H
#define ANGULAR_VELOCITY_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real get_angular_speed()
{
    Real omega = 0.0_rt;


    if (hydrosim::spin_period > 0.0_rt) {
        omega = 2.0_rt * std::numbers::pi / hydrosim::spin_period;
    }

    return omega;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
GpuArray<Real, 3> get_angular_velocity(const GeometryData& geomdata,
                                const int j)
{
    GpuArray<Real, 3> angular_velocity_vec = {0.0_rt};

    const auto coord = geomdata.Coord();
    const auto problo = geomdata.ProbLo(1);
    const auto theta_cell_width = geomdata.CellSize(1);

    auto omega = get_angular_speed();

    if (coord == 2) {
        Real theta;
        if (hydrosim::fixed_latitude_angle >= 0.0_rt) {
            theta = hydrosim::fixed_latitude_angle;
        } else {
            theta = problo + (static_cast<Real>(j) + 0.5_rt) * theta_cell_width;
        }
        angular_velocity_vec[0] = std::cos(theta) * omega;
        angular_velocity_vec[1] = -std::sin(theta) * omega;
    } else {
        angular_velocity_vec[hydrosim::spin_axis - 1] = omega;
    }

    return angular_velocity_vec;
}

#endif
