# subch_planar

This setup studies the behavior of the initial detonation of a double
detonation Type Ia supernova at the white dwarf and accretion layer
interface and the effect of prohibiting burning in detected shock
zones. It takes a 1-d initial model representing a sub-Chandra WD and
a surface He layer and initializes a small perturbation in the He
layer.

To disable burning in shocks:

* in the inputs file, set `hydrosim.disable_shock_burning = 1`

* to change the detection threshold from the default 2/3, use
  `hydrosim.shock_detection_threshold = #`, where # is the new
  desired threshold.
