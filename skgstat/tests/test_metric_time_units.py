import unittest

import numpy as np
from numpy.testing import assert_allclose

from skgstat import SpatioTemporalVariogram


class TestMetricTimeUnits(unittest.TestCase):
    @staticmethod
    def exact_surface(factor):
        x, t = np.meshgrid(np.arange(20.) * 3, np.arange(21.) / 5,
                           indexing='ij')
        c = np.column_stack((x.ravel(), t.ravel() * factor))
        values = np.sin(x.ravel() / 10) + np.cos(t.ravel())
        var = SpatioTemporalVariogram(
            c, values, model='metric', marginal_model='exponential',
            use_nugget=True, xbins=np.array([0., 3., 6., 12., 24., 40., 60.]),
            tbins=np.array([0., .2, .4, .8, 1.5, 2.5, 4.]) * factor)
        h, dt = var.meshbins
        # Independent analytic exponential surface, only at occupied lags.
        radius = np.hypot(h / 30, dt / (2 * factor))
        exact = .2 + 1.7 * -np.expm1(-3 * radius)
        var._experimental = np.where(np.isfinite(var.experimental), exact, np.nan)
        return var

    def test_seconds_for_multi_year_time_range(self):
        for factor in [1., 86400., 31557600.]:
            var = self.exact_surface(factor)
            var.fit()
            assert_allclose(var.cof / [30., 2 * factor, 1.7, .2], 1., rtol=1e-4)

    def test_covariance_uses_parameter_units(self):
        factor = 31557600.
        original = self.exact_surface(1.)
        scaled = self.exact_surface(factor)
        original.fit()
        scaled.fit()
        scale = np.array([1., factor, 1., 1.])
        assert_allclose(scaled.cov / np.outer(scale, scale), original.cov,
                        rtol=.02, atol=1e-22)

    def test_constant_surface_still_rejected(self):
        var = self.exact_surface(1.)
        var._experimental[:] = 0.
        with self.assertRaises(ValueError):
            var.fit()


if __name__ == '__main__':
    unittest.main()
