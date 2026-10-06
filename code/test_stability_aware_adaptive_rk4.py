import unittest

import numpy as np

import stability_aware_adaptive_rk4 as experiment


class StabilityAwareTests(unittest.TestCase):
    def test_reduced_jacobian_matches_finite_difference(self):
        y = np.array([0.7, 2.0e-5, 1.0 - 0.7 - 2.0e-5])
        epsilon = 1.0e-6

        def reduced_rhs(reduced_y):
            full_y = np.array(
                [reduced_y[0], reduced_y[1], 1.0 - reduced_y.sum()]
            )
            return experiment.rhs(0.0, full_y)[:2]

        finite_difference = np.column_stack(
            [
                (
                    reduced_rhs(y[:2] + epsilon * np.eye(2)[column])
                    - reduced_rhs(y[:2] - epsilon * np.eye(2)[column])
                )
                / (2.0 * epsilon)
                for column in range(2)
            ]
        )
        np.testing.assert_allclose(
            experiment.reduced_jacobian(y), finite_difference, rtol=1.0e-8, atol=1.0e-7
        )

    def test_stability_cap_is_on_rk4_boundary_when_below_hmax(self):
        y = np.array([0.5, 1.0e-5, 0.49999])
        cap = experiment.local_stability_cap(y)
        eigenvalues = np.linalg.eigvals(experiment.reduced_jacobian(y))
        self.assertTrue(experiment.outer_step_is_stable(cap, eigenvalues))
        if cap < experiment.H_MAX * (1.0 - 1.0e-10):
            self.assertFalse(
                experiment.outer_step_is_stable(cap * (1.0 + 1.0e-8), eigenvalues)
            )

    def test_stability_aware_run_never_exceeds_safe_cap(self):
        run = experiment.integrate(1.0e-4, stability_aware=True)
        self.assertLessEqual(
            np.max(run.steps / run.stability_caps),
            experiment.STABILITY_SAFETY * (1.0 + 1.0e-12),
        )
        self.assertTrue(run.summary.reached_t40)


if __name__ == "__main__":
    unittest.main()
