"""Vendored DBO exercises the installed Rust-backed package with real modules."""

import importlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from test_dbo_runtime import BACKEND_DIR, HAS_REAL_STACK, restore_real_modules


@unittest.skipUnless(HAS_REAL_STACK, "requires torch, botorch-rust and gpytorch")
class DboPortIntegrationTests(unittest.TestCase):
    def setUp(self):
        restore_real_modules()
        if str(BACKEND_DIR) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIR))
        self.torch = importlib.import_module("torch")
        self.dbo = importlib.import_module("dbo_torch")
        self.model_module = importlib.import_module("dbo_torch.model")
        self.base_module = importlib.import_module("dbo_torch._base")
        threads = self.torch.get_num_threads()
        self.torch.set_num_threads(1)
        self.addCleanup(self.torch.set_num_threads, threads)

    def data(self, device="cpu"):
        X = self.torch.tensor(
            [[0.1, 1.0], [0.35, 2.0], [0.65, 3.0], [0.9, 4.0]],
            dtype=self.torch.double,
            device=device,
        )
        return X, (3 * X[:, :1]).sin()

    def test_cooperative_model_preserves_actual_kernel_likelihood_and_decay_graph(self):
        from botorch_rs.acquisition import LogExpectedImprovement
        from botorch_rs.mlls import ExactMarginalLogLikelihood
        from botorch_rs.models import SingleTaskGP
        from gpytorch.kernels import RBFKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood

        devices = ["cpu"] + (["cuda"] if self.torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                X, Y = self.data(device)
                temporal = self.dbo.TemporalDecayKernel(initial_alpha=0.8, active_dims=[1]).to(X)
                kernel = (ScaleKernel(RBFKernel(active_dims=[0])) * temporal).to(X)
                likelihood = GaussianLikelihood().to(X)
                likelihood.noise = 0.04
                alpha = temporal.raw_alpha
                model = self.model_module._DBOSingleTaskGP(
                    X,
                    Y,
                    covar_module=kernel,
                    likelihood=likelihood,
                    outcome_transform=None,
                    time_dim=1,
                    backend="torch",
                )
                self.assertIsInstance(model, SingleTaskGP)
                self.assertIsInstance(model, self.torch.nn.Module)
                self.assertIs(model.covar_module, kernel)
                self.assertIs(model.likelihood, likelihood)
                self.assertIs(self.model_module._temporal_kernel(model).raw_alpha, alpha)
                self.assertEqual(model._ignore_X_dims_scaling_check, [1])
                self.assertEqual(model._tensor_backend, "torch")
                objective = ExactMarginalLogLikelihood(likelihood, model)

                def first_derivative():
                    value = -objective()
                    return self.torch.autograd.grad(value, alpha, create_graph=True)[0]

                gradient = first_derivative()
                higher = self.torch.autograd.grad(gradient.sum(), alpha)[0]
                original, step, sides = alpha.detach().clone(), 1e-5, []
                for sign in (1, -1):
                    with self.torch.no_grad():
                        alpha.copy_(original + sign * step)
                    sides.append(first_derivative().detach())
                with self.torch.no_grad():
                    alpha.copy_(original)
                self.torch.testing.assert_close(
                    higher, (sides[0] - sides[1]) / (2 * step), atol=2e-7, rtol=2e-6
                )
                self.assertGreater(float(gradient.detach().abs().sum()), 0)
                query = X.new_tensor([[[0.25, 5.0]], [[0.75, 5.5]]], requires_grad=True)
                acquisition = LogExpectedImprovement(model, best_f=0.1, maximize=False)
                value = acquisition(query)
                query_gradient, decay_gradient = self.torch.autograd.grad(
                    value.sum(), (query, alpha)
                )
                self.assertTrue(bool(self.torch.isfinite(query_gradient).all()))
                self.assertTrue(bool(self.torch.isfinite(decay_gradient).all()))

    def test_multistart_selects_actual_highest_likelihood_state(self):
        X, Y = self.data()
        model = self.model_module.build_model(
            X,
            Y,
            self.dbo.DBOModelConfig(lengthscale_starts=(1.0,)),
            bounds=X.new_tensor([[0.0], [1.0]]),
        )
        starting = self.model_module._spatial_kernel(model).lengthscale.detach().clone()
        visited, scores, states = [], [], []

        def numerical_fit(mll):
            fitted = mll.model
            visited.append(self.model_module._spatial_kernel(fitted).lengthscale.detach().clone())
            scale = fitted.covar_module.kernels[0]
            scale.outputscale = 0.2 + len(visited) * 0.35
            self.model_module._temporal_kernel(fitted).alpha = 0.6 + len(visited) * 0.07
            states.append(self.model_module._state(fitted))
            scores.append(self.model_module._total_log_likelihood(fitted))
            return mll

        with mock.patch.object(self.model_module, "fit_gpytorch_mll", numerical_fit):
            self.model_module.fit_model(model, lengthscale_starts=(1.0, 0.5, 0.25))
        for observed, factor in zip(visited, (1.0, 0.5, 0.25), strict=True):
            self.torch.testing.assert_close(observed, starting * factor)
        selected = scores.index(max(scores))
        for name, actual in model.state_dict().items():
            self.torch.testing.assert_close(actual, states[selected][name])
        self.assertAlmostEqual(self.model_module._total_log_likelihood(model), max(scores))

    def test_failed_fit_restores_mutable_state_and_retries_with_raised_noise_floor(self):
        X, Y = self.data()
        model = self.model_module.build_model(X, Y)
        kernel = model.covar_module.kernels[0]
        initial_scale = kernel.outputscale.detach().clone()
        initial_floor = float(model.likelihood.noise_covar.raw_noise_constraint.lower_bound)
        observed = []

        def numerical_fit(mll):
            observed.append(
                (
                    kernel.outputscale.detach().clone(),
                    float(mll.likelihood.noise_covar.raw_noise_constraint.lower_bound),
                )
            )
            if len(observed) == 1:
                kernel.outputscale = 7.0
                self.model_module._temporal_kernel(model).alpha = 0.2
                raise RuntimeError("simulated failed numerical fit")
            return mll

        with mock.patch.object(self.model_module, "fit_gpytorch_mll", numerical_fit):
            result = self.model_module.fit_model(model, max_attempts=2)
        self.assertIs(result, model)
        self.assertEqual(len(observed), 2)
        self.torch.testing.assert_close(observed[1][0], initial_scale)
        self.assertAlmostEqual(observed[1][1], 2 * initial_floor)
        self.assertGreater(float(model.likelihood.noise), observed[1][1])

    def test_warm_start_carries_parameters_without_reusing_data_or_transform_statistics(self):
        config = self.dbo.DBOConfig(warm_start=True, refit_every=2, seed=7)
        optimizer = self.dbo.DynamicBO([(0.0, 1.0)], config)
        X, Y = self.data()
        previous = self.model_module.build_model(X, Y, bounds=optimizer.bounds)
        self.model_module._temporal_kernel(previous).alpha = 0.73
        previous.covar_module.kernels[0].outputscale = 1.4
        larger_X = self.torch.cat((X, X.new_tensor([[0.5, 5.0]])))
        larger_Y = self.torch.cat((Y, Y.new_tensor([[1.7]])))
        optimizer.observe([0.5], 0.3)
        with mock.patch.object(self.base_module, "fit_model", side_effect=AssertionError("refit")):
            carried = optimizer._refresh_model(larger_X, larger_Y, previous)
        self.assertIs(carried._train_X, larger_X)
        self.assertIsNot(carried.covar_module, previous.covar_module)
        self.assertAlmostEqual(self.model_module.get_alpha(carried), 0.73)
        self.torch.testing.assert_close(
            carried.covar_module.kernels[0].outputscale,
            previous.covar_module.kernels[0].outputscale,
        )
        self.torch.testing.assert_close(
            carried.outcome_transform.means, larger_Y.mean(-2, keepdim=True)
        )
        self.assertFalse(
            self.torch.equal(carried.outcome_transform.means, previous.outcome_transform.means)
        )

    def test_dynamic_mobo_qlognehvi_validation_and_save_resume_smoke(self):
        from botorch_rs.models import ModelListGP

        model_config = self.dbo.DBOModelConfig(lengthscale_starts=(1.0,))
        config = self.dbo.MODBOConfig(
            model=model_config,
            seed_points=[[0.1], [0.5], [0.9]],
            seed=19,
            num_restarts=2,
            raw_samples=16,
            mc_samples=8,
            validation_every=5,
        )
        optimizer = self.dbo.DynamicMOBO([(0.0, 1.0)], [2.0, 2.0], config)
        for iteration in range(4):
            x = optimizer.suggest()
            self.assertTrue(0 <= x[0] <= 1)
            optimizer.observe(
                x,
                [
                    (x[0] - 0.2 - 0.01 * iteration) ** 2,
                    (x[0] - 0.8 + 0.01 * iteration) ** 2,
                ],
            )
        self.assertIsInstance(optimizer._model_list, ModelListGP)
        self.assertEqual(optimizer._model_list._tensor_backend, "torch")
        self.assertEqual(len(optimizer.alphas), 2)
        self.assertIsNot(
            self.model_module._temporal_kernel(optimizer._models[0]).raw_alpha,
            self.model_module._temporal_kernel(optimizer._models[1]).raw_alpha,
        )
        self.assertTrue(optimizer.pareto_front())
        trace = optimizer.hypervolume_trace()
        self.assertEqual(len(trace), 4)
        self.assertEqual(trace, sorted(trace))
        with tempfile.TemporaryDirectory() as temporary:
            state = optimizer.save(pathlib.Path(temporary) / "mo.json")
            payload = json.loads(state.read_text())
            self.assertIn("botorch-rust", payload["versions"])
            self.assertNotIn("botorch", payload["versions"])
            restored = self.dbo.DynamicMOBO.load(state)
            expected = optimizer.suggest()
            self.assertEqual(restored.suggest(), expected)
            observation = restored.observe(expected, [0.3, 0.4])
            self.assertTrue(observation.is_validation)
            self.assertEqual(len(observation.predicted_y), 2)
            self.assertEqual(len(observation.predicted_sd), 2)


if __name__ == "__main__":
    unittest.main()
