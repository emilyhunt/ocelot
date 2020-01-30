"""A set of constants calculated using astropy.units once at runtime."""

from astropy import units as u

# Constants we'll need
mas_per_yr_to_rad_per_s = (u.mas / u.yr).to(u.rad / u.s)
deg_to_rad = u.deg.to(u.rad)
pc_to_m = u.parsec.to(u.meter)
