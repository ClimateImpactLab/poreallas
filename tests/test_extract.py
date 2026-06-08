"""
Test logic related to region extraction and transformation
"""
import isku
import numpy as np
import pytest
import xarray as xr

from poreallas.extract import (
    _make_annual_tas,
    _make_30hbartlett_climtas,
    make_climtas,
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

