import numpy as np
from ocelot.model.binaries import MoeDiStefanoMultiplicityRelation
from ocelot.model.binaries.moe_distefano_17 import (
    mass,
    multiplicity_fraction,
    companion_star_frequency,
)


def test_mass_ratio_distribution():
    result = MoeDiStefanoMultiplicityRelation().multiplicity_fraction(mass)
    assert isinstance(result, np.ndarray)
    assert result.shape == mass.shape
    np.testing.assert_allclose(result, multiplicity_fraction)


def test_companion_star_frequency():
    result = MoeDiStefanoMultiplicityRelation().companion_star_frequency(mass)
    assert isinstance(result, np.ndarray)
    assert result.shape == mass.shape
    np.testing.assert_allclose(result, companion_star_frequency)


def test_random_mass_ratio():
    masses = np.geomspace(0.01, 1000, num=500)
    result = MoeDiStefanoMultiplicityRelation().random_mass_ratio(masses)
    assert isinstance(result, np.ndarray)
    assert result.shape == masses.shape
    assert np.all(result > 0.0)
    assert np.all(result <= 1.0)
    # Todo see if we can test against paper values somehow? Probably not easy...


def test_random_mass_ratio_seeded():
    np.random.seed(42)  # Todo remove - due to imf.distribution issue
    result = MoeDiStefanoMultiplicityRelation().random_mass_ratio(mass, seed=42)
    assert isinstance(result, np.ndarray)
    assert result.shape == mass.shape

    expected_result = np.asarray(
        [0.943944, 0.669913, 0.358638, 0.265084, 0.989303, 0.203655, 0.288664]
    )
    np.testing.assert_allclose(result, expected_result, atol=1e-6, rtol=1e-5)


def test_random_binary():
    masses = np.geomspace(0.01, 1000, num=500)
    mass_ratios, periods, eccentricities = (
        MoeDiStefanoMultiplicityRelation().random_binary(masses)
    )
    for result in (mass_ratios, periods, eccentricities):
        assert isinstance(result, np.ndarray)
        assert result.shape == masses.shape
    assert np.all(mass_ratios > 0.0)
    assert np.all(mass_ratios <= 1.0)
    assert np.all(periods > 0.0)
    assert np.all(periods <= 10**8)
    assert np.all(eccentricities >= 0.0)
    assert np.all(eccentricities < 1.0)


def test_random_binary_seeded():
    np.random.seed(42)  # Todo remove - due to imf.distribution issue
    mass_ratios, periods, eccentricities = (
        MoeDiStefanoMultiplicityRelation().random_binary(mass, seed=42)
    )

    expected_mass_ratios = np.asarray(
        [0.943944, 0.669913, 0.358638, 0.265084, 0.989303, 0.203655, 0.288664]
    )
    expected_periods = np.asarray(
        [
            3.175529e06,
            3.923213e04,
            1.128245e07,
            3.389371e05,
            1.206090e01,
            5.050596e07,
            3.466024e05,
        ]
    )
    expected_eccentricities = np.asarray(
        [0.877159, 0.330149, 0.295915, 0.322699, 0.331873, 0.698901,
                  0.627076]
    )
    np.testing.assert_allclose(mass_ratios, expected_mass_ratios, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(periods, expected_periods, atol=10, rtol=1e-5)
    np.testing.assert_allclose(
        eccentricities, expected_eccentricities, atol=1e-6, rtol=1e-5
    )


def test_random_binary_one_binary():
    result = MoeDiStefanoMultiplicityRelation().random_binary(np.asarray([0.1]), seed=42)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[2], np.ndarray)
    assert result[0].shape == (1,)
    assert result[1].shape == (1,)
    assert result[2].shape == (1,)


def test_random_binary_against_paper_values():
    """Uses values loosely read off of Table 13 in the paper."""
    n_samples = 3000
    relation = MoeDiStefanoMultiplicityRelation()
    result_1_msun = relation.random_binary(np.full(n_samples, 1.0))
    result_3p5_msun = relation.random_binary(np.full(n_samples, 3.5))
    result_7_msun = relation.random_binary( np.full(n_samples, 7.0))
    result_12p5_msun = relation.random_binary(np.full(n_samples, 12.5))
    result_20_msun = relation.random_binary(np.full(n_samples, 20))
    results = result_1_msun, result_3p5_msun, result_7_msun, result_12p5_msun, result_20_msun


    expected_periods = [4.9, 4.2, 3.9, 3.75, 3.6]

    for expected_period, result in zip(expected_periods, results):
        np.testing.assert_allclose(expected_period, np.median(np.log10(result[1])), rtol=0.1)


# def test_many():
#     for i in range(100):
#         test_random_binary_against_paper_values()
