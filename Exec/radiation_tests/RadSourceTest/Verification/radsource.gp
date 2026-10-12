set term post eps color enhanced
set output 'radiating_source.eps'

set xlabel 'time (s)'
set ylabel '{/Symbol r}e (erg/cm^3)'

set format "10^{%L}"

set logscale

set yrange [30:3.e9]

set key left center

plot 'analytic_heating.out' using 1:2 title 'analytic heating solution' w l, \
     'hydrosim_heating.out' using 1:2 title 'HydroSim heating solution', \
     'analytic_cooling.out' using 1:2 title 'analytic cooling solution' w l, \
     'hydrosim_cooling.out' using 1:2 title 'HydroSim cooling solution'

