import numpy as np

from sk_gp.envs import compute_grid, environment_from_scene
from sk_gp.kernel import (
    SemanticGibbsRBF,
    SemanticHyperField,
    SemanticKernel,
    SemanticLengthscale,
    build_gate_entities,
    resolve_gate_material_params,
)
from sk_gp.simulator import generate_scene


def _small_env():
    scene = generate_scene(seed=2, complexity="simple")
    return environment_from_scene(scene.to_json(hide_materials=False, hide_ap=False))


def test_semantic_gibbs_rbf_train_gram_is_psd():
    env = _small_env()
    sem_l = SemanticLengthscale(env, tau=1.0, beta=0.2, l_min=0.4, l_max=1.4, radius=2.0)
    kernel = SemanticGibbsRBF(sem_l, variance=1.0, jitter=1e-4)
    X = np.random.default_rng(0).uniform(0, 4, size=(20, 2))
    K = kernel(X)
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T, atol=1e-8)
    np.linalg.cholesky(K)  # raises if not PD


def test_semantic_kernel_gate_bounded_and_psd():
    env = _small_env()
    sem_l = SemanticLengthscale(env, tau=1.0, beta=0.2, l_min=0.4, l_max=1.4, radius=2.0)
    hf = SemanticHyperField(sem_l, K=3, temperature=0.85, center_mode="linear", priors=[0.5, 1.0, 0.7])
    entities = build_gate_entities(env, resolve_gate_material_params(env, {}))
    kernel = SemanticKernel(hyperfield=hf, gate_entities=entities, gate_distance_scale=1.0, variance=1.0, jitter=1e-4)

    X = np.random.default_rng(1).uniform(0, 4, size=(25, 2))
    Y = np.random.default_rng(2).uniform(0, 4, size=(10, 2))
    K_cross = kernel(X, Y)
    assert K_cross.shape == (25, 10)

    # Semantic + geometry gate keeps the cross-covariance magnitude <= variance.
    assert np.all(K_cross <= 1.0 + 1e-6)

    K_train = kernel(X)
    assert np.allclose(K_train, K_train.T, atol=1e-8)
    np.linalg.cholesky(K_train)


def test_noise_field_adds_to_diagonal():
    from sk_gp.kernel.fields import NoiseField

    env = _small_env()
    sem_l = SemanticLengthscale(env, tau=1.0, beta=0.2, l_min=0.4, l_max=1.4, radius=2.0)
    hf = SemanticHyperField(sem_l, K=2, center_mode="linear")
    noise = NoiseField([[1, 1]], [0.5], tau=10.0, radius=1e6)
    kernel = SemanticKernel(hyperfield=hf, gate_entities=[], gate_distance_scale=1.0, variance=1.0, noise_field=noise)
    X = np.random.default_rng(3).uniform(0, 4, size=(8, 2))
    diag_no_noise = np.diag(SemanticKernel(hyperfield=hf, gate_entities=[], gate_distance_scale=1.0, variance=1.0)(X))
    diag_noise = np.diag(kernel(X))
    assert np.all(diag_noise > diag_no_noise)
