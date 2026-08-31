"""
Test logic related to region extraction and transformation

These are adapted from the tests in the prototype https://github.com/brews/ideal-succotash/blob/c30a9d8226aaef806a16c5fdae3c186761811829/tests/mortality/test_transformation.py.
"""

import isku
import numpy as np
import pytest
import xarray as xr

from poreallas.extract import (
    _make_annual_tas,
    _make_30hbartlett_climtas,
    make_climtas,
    make_tas_monthly_histogram,
)


@pytest.fixture
def basic_segment_weights():
    sw = isku.GridWeightingRegions(
        weights=xr.Dataset(
            {
                "region": (["idx"], ["foobar"]),
                "weight": (["idx"], [1.0]),
                "lon": (["idx"], [1.0]),
                "lat": (["idx"], [0.0]),
            },
        )
    )
    return sw


def test__make_annual_tas():
    """
    Test that _make_annual_tas grabs "tas" variable from a Dataset and spits out
    a Dataset with time averaged in a new "year" dim.

    This covers the new "year" dim moving to the first dimension but I'm not sure that matters.
    """
    expected = xr.Dataset(
        {"tas": (["lon", "lat", "year"], [[[-91.15, 91.85]]])},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "year": [2023, 2024],
        },
    )

    ds_in = xr.Dataset(
        {"tas": (["lon", "lat", "time"], np.arange(366).reshape((1, 1, 366)))},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range("2023-01-01", "2024-01-01", freq="1D"),
        },
    )
    ds_in["tas"].attrs["units"] = "degK"

    actual = _make_annual_tas(ds_in)
    xr.testing.assert_allclose(actual, expected)


def test__make_30hbartlett_climtas():
    """
    Test _make_30hbartlett_climtas creates a 30 year half-Bartlett average
    returned as "climtas".
    """
    ex = np.empty((31, 1, 1), dtype=np.float32)
    ex[:] = np.nan
    ex[-2, ...] = 19.666666
    ex[-1, ...] = 20.666666

    expected = xr.Dataset(
        {"climtas": (["year", "lon", "lat"], ex.reshape(31, 1, 1))},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "year": np.arange(2000, 2031),
        },
    )

    ds_in = xr.Dataset(
        {"tas": (["year", "lon", "lat"], np.arange(31).reshape((31, 1, 1)))},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "year": np.arange(2000, 2031),
        },
    )

    actual = _make_30hbartlett_climtas(ds_in)
    xr.testing.assert_allclose(actual, expected)


def test_make_climtas(basic_segment_weights):
    """
    Test that make_climtas transformation runs through apply_transformation using basic_segment_weights without error, spitting out smoothed annual "climtas" variables from input daily "tas".
    """
    ex = np.empty((1, 31), dtype=np.float32)
    ex[:] = np.nan
    ex[..., -2] = 7360.3335
    ex[..., -1] = 7725.3335
    expected = xr.Dataset(
        {"climtas": (["region", "year"], ex.reshape(1, 31))},
        coords={
            "region": ["foobar"],
            "year": np.arange(2000, 2031),
        },
    )

    ds_in = xr.Dataset(
        {
            "tas": (
                ["lon", "lat", "time"],
                np.arange(11315, dtype=np.float32).reshape(1, 1, 11315),
            )
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range(
                "2000-01-01", "2030-12-31", freq="1D", calendar="noleap"
            ),
        },
    )
    ds_in["tas"].attrs["units"] = "degC"

    actual = isku.extract_regions(
        ds_in,
        template=make_climtas,
        regions=basic_segment_weights,
    )
    xr.testing.assert_allclose(actual, expected)


def test_make_monthly_histogram(basic_segment_weights):
    """
    Test that make_tas_monthly_histogram extraction run through apply_transformation.

    This basically tests that it runs, the output is a histogram monthly time series when given a daily tas timeseries as input.
    """

    # Building the expected output.
    # Input is just 1.0 repeated for 365 days in the year 2000. If these are
    # broken down into monthly histograms, we expect the histogram bin covering "1.0"
    # to have a count equal to the number of days for that month.
    # So, we first build a Dataset with the needed structure. Then, we insert the
    # count of "days_in_month" across the time slice, for the index of the bin containing 1.0.
    expected = xr.Dataset(
        {"histogram_tas": (["region", "time", "tas_bin"], np.zeros((1, 12, 170)))},
        coords={
            "region": np.array(["foobar"]),
            "time": xr.date_range(
                "2000-01-01", "2000-12-31", freq="1MS", calendar="noleap"
            ),
            "tas_bin": np.arange(-105, 65) + 0.5,
        },
    )
    expected["histogram_tas"].data[0, :, 106] = expected["time.days_in_month"].data

    ds_in = xr.Dataset(
        {
            "tas": (
                ["lon", "lat", "time"],
                np.ones(365, dtype=np.float32).reshape(1, 1, 365),
            )
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range(
                "2000-01-01", "2000-12-31", freq="1D", calendar="noleap"
            ),
        },
    )
    ds_in["tas"].attrs["units"] = "degC"

    actual = isku.extract_regions(
        ds_in, template=make_tas_monthly_histogram, regions=basic_segment_weights
    )

    xr.testing.assert_allclose(actual, expected)
