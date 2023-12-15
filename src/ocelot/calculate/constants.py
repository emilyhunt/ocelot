"""A set of constants calculated using astropy.units once at runtime."""

from astropy import units as u

# Wavelengths of the Gaia DR2 photometry in nanometres
gaia_dr2_wavelengths = {"G": 643.770, "G_BP": 530.957, "G_RP": 770.985}

# A_lambda over A_V for Gaia DR2
gaia_dr2_a_lambda_over_a_v = {"G": 0.85926, "G_BP": 1.06794, "G_RP": 0.65199}

# Gaia DR2 photometric information from Evans+18:
gaia_dr2_zero_points = {"G": 25.6884, "G_BP": 25.3514, "G_RP": 24.7619}
gaia_dr2_zero_points_error = {"G": 0.0018, "G_BP": 0.0014, "G_RP": 0.0019}


# Constants we'll need
mas_per_yr_to_rad_per_s = (u.mas / u.yr).to(u.rad / u.s)
deg_to_rad = u.deg.to(u.rad)
pc_to_m = u.parsec.to(u.meter)

# Default key names for anything we calculate in the module
# Todo: propagate this dict into all the functions below
default_ocelot_key_names = {
    # Position
    "name": "name",
    "ra": "ra",
    "ra_error": "ra_error",
    "dec": "dec",
    "dec_error": "dec_error",
    # Angular size
    "ang_radius_50": "ang_radius_50",
    "ang_radius_50_error": "ang_radius_50_error",
    "ang_radius_c": "ang_radius_c",
    "ang_radius_c_error": "ang_radius_c_error",
    "ang_radius_t": "ang_radius_t",
    "ang_radius_t_error": "ang_radius_t_error",
    # Physical size
    "radius_50": "ang_radius_50",
    "radius_50_error": "ang_radius_50_error",
    "radius_c": "ang_radius_c",
    "radius_c_error": "ang_radius_c_error",
    "radius_t": "ang_radius_t",
    "radius_t_error": "ang_radius_t_error",
    # Distance
    "parallax": "parallax",
    "parallax_error": "parallax_error",
    "inverse_parallax": "inverse_parallax",
    "inverse_parallax_l68": "inverse_parallax_l68",
    "inverse_parallax_u68": "inverse_parallax_u68",
    "distance": "distance",
    "distance_error": "distance_error",
    # Proper motion and velocity
    "pmra": "pmra",
    "pmra_error": "pmra_error",
    "pmdec": "pmdec",
    "pmdec_error": "pmdec_error",
    "v_internal_tangential": "v_internal_tangential",
    "v_internal_tangential_error": "v_internal_tangential_error",
    # Diagnostics
    "parameter_inference_mode": "parameter_inference_mode",
}
