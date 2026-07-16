# context_support.py — shared contextual-BO support for bo.py and mobo.py.
#
# Implements the optional "context" section of the Unity init protocol and the
# LCE-M GP (latent context embedding multi-task GP, Feng et al., NeurIPS 2020)
# model construction on top of BoTorch's ``LCEMGP``.
#
# Context embeddings are definable in three ways ("embeddingSource"):
#   * "learned": a low-dimensional embedding per context is learned from data
#     (LCEMGP default behaviour, no user-provided features).
#   * "manual":  each context carries a user-provided embedding vector
#     (e.g. pre-computed with any encoder of your choice).
#   * "image":   each context carries an image; the Python backend embeds it
#     with an open_clip vision transformer (default: ViT-bigG-14, the open_clip
#     release of ViT-G/14). Requires the optional dependency ``open_clip_torch``.
#
# Only numpy / stdlib are imported at module scope so this file stays importable
# in test environments without torch/botorch installed. torch and botorch are
# imported lazily inside the functions that need them.

import hashlib
import os

import numpy as np

VALID_EMBEDDING_SOURCES = ("learned", "manual", "image")

# open_clip model name of ViT-G/14 and the matching pretrained weight tag.
DEFAULT_IMAGE_EMBEDDING_MODEL = "ViT-bigG-14"
DEFAULT_IMAGE_EMBEDDING_PRETRAINED = "laion2b_s39b_b160k"

# Column name used for context keys in warm-start parameter CSVs and in
# ObservationsPerEvaluation.csv when contextual optimization is enabled.
CONTEXT_CSV_COLUMN = "Context"


class ContextSetup:
    """Validated contextual-optimization configuration.

    Context index = position of the context key in ``keys`` (0..K-1); the same
    index is used as the task feature value of the multi-task GP.
    """

    def __init__(self, keys, current_key, embedding_source, normalize_embeddings,
                 manual_embeddings=None, image_paths=None,
                 image_model=DEFAULT_IMAGE_EMBEDDING_MODEL,
                 image_pretrained=DEFAULT_IMAGE_EMBEDDING_PRETRAINED):
        self.keys = list(keys)
        self.current_key = current_key
        self.current_index = self.keys.index(current_key)
        self.embedding_source = embedding_source
        self.normalize_embeddings = bool(normalize_embeddings)
        self.manual_embeddings = manual_embeddings  # (K, m) ndarray or None
        self.image_paths = image_paths              # list[str] or None
        self.image_model = image_model
        self.image_pretrained = image_pretrained
        # Filled by resolve_embeddings(): (K, m) float64 ndarray or None.
        self.embeddings = None
        self.embeddings_resolved = False

    @property
    def num_contexts(self):
        return len(self.keys)

    def index_of(self, key):
        try:
            return self.keys.index(key)
        except ValueError:
            raise ValueError(
                f"Unknown context key '{key}'. Configured contexts: {self.keys}"
            ) from None


def parse_context_config(init_msg):
    """Parse and validate the optional "context" section of the init message.

    Returns a ContextSetup, or None when contextual optimization is disabled.
    Embeddings are not resolved yet; call resolve_embeddings() afterwards.
    """
    ctx = (init_msg or {}).get("context") or {}
    if not isinstance(ctx, dict):
        raise ValueError(f"init 'context' must be an object, got {type(ctx).__name__}")
    if not bool(ctx.get("enabled", False)):
        return None

    source = str(ctx.get("embeddingSource", "learned") or "learned").strip().lower()
    if source not in VALID_EMBEDDING_SOURCES:
        raise ValueError(
            f"context.embeddingSource must be one of {list(VALID_EMBEDDING_SOURCES)}, got '{source}'"
        )

    entries = ctx.get("contexts") or []
    if not isinstance(entries, list) or len(entries) < 1:
        raise ValueError("context.contexts must be a non-empty list when contextual optimization is enabled.")

    keys = []
    seen_lower = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"context.contexts[{i}] must be an object, got {type(entry).__name__}")
        key = str(entry.get("key") or "").strip()
        if not key:
            raise ValueError(f"context.contexts[{i}] has an empty key.")
        if key.lower() in seen_lower:
            raise ValueError(f"Duplicate context key '{key}' (context keys must be unique, case-insensitively).")
        seen_lower.add(key.lower())
        keys.append(key)

    if len(keys) == 1:
        print(
            "Warning: only one context is configured. Contextual optimization is most useful "
            "with multiple contexts (e.g. warm-start data from other contexts).",
            flush=True,
        )

    current_key = str(ctx.get("currentContext") or "").strip()
    if not current_key:
        raise ValueError("context.currentContext must be set when contextual optimization is enabled.")
    matched_current = [k for k in keys if k.lower() == current_key.lower()]
    if not matched_current:
        raise ValueError(
            f"context.currentContext '{current_key}' is not in the configured context list {keys}."
        )
    current_key = matched_current[0]

    normalize_embeddings = bool(ctx.get("normalizeEmbeddings", True))

    manual_embeddings = None
    image_paths = None
    if source == "manual":
        rows = []
        dim = None
        for i, entry in enumerate(entries):
            emb = entry.get("embedding")
            if not isinstance(emb, (list, tuple)) or len(emb) < 1:
                raise ValueError(
                    f"context '{keys[i]}' must provide a non-empty 'embedding' vector for embeddingSource=manual."
                )
            if dim is None:
                dim = len(emb)
            elif len(emb) != dim:
                raise ValueError(
                    f"context embedding length mismatch: context '{keys[i]}' has {len(emb)} values, expected {dim}."
                )
            try:
                row = np.asarray([float(v) for v in emb], dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise ValueError(f"context '{keys[i]}' embedding contains non-numeric values.") from e
            if not np.all(np.isfinite(row)):
                raise ValueError(f"context '{keys[i]}' embedding contains NaN/Inf values.")
            rows.append(row)
        manual_embeddings = np.stack(rows, axis=0)
    elif source == "image":
        image_paths = []
        for i, entry in enumerate(entries):
            path = str(entry.get("imagePath") or "").strip()
            if not path:
                raise ValueError(
                    f"context '{keys[i]}' must provide an 'imagePath' for embeddingSource=image."
                )
            image_paths.append(path)

    image_model = str(ctx.get("imageEmbeddingModel") or DEFAULT_IMAGE_EMBEDDING_MODEL).strip()
    image_pretrained = str(ctx.get("imageEmbeddingPretrained") or DEFAULT_IMAGE_EMBEDDING_PRETRAINED).strip()

    return ContextSetup(
        keys=keys,
        current_key=current_key,
        embedding_source=source,
        normalize_embeddings=normalize_embeddings,
        manual_embeddings=manual_embeddings,
        image_paths=image_paths,
        image_model=image_model,
        image_pretrained=image_pretrained,
    )


def l2_normalize_rows(arr):
    """L2-normalize each row; zero rows are left untouched (with a warning)."""
    arr = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    zero_rows = norms[:, 0] <= 1e-12
    if np.any(zero_rows):
        print(
            f"Warning: {int(np.sum(zero_rows))} context embedding(s) have (near-)zero norm and were not normalized.",
            flush=True,
        )
    safe_norms = np.where(norms <= 1e-12, 1.0, norms)
    return arr / safe_norms


def resolve_embeddings(setup, init_root=None):
    """Resolve the (K, m) embedding matrix for the given ContextSetup in-place.

    * learned: embeddings stay None (LCEMGP learns them from data).
    * manual:  uses the user-provided vectors.
    * image:   embeds each context image with the configured vision model.

    Relative image paths are resolved against ``init_root`` (the InitData
    folder Unity ships alongside warm-start CSVs).
    """
    if setup is None:
        return None

    if setup.embedding_source == "learned":
        setup.embeddings = None
    elif setup.embedding_source == "manual":
        setup.embeddings = np.asarray(setup.manual_embeddings, dtype=np.float64)
    elif setup.embedding_source == "image":
        paths = []
        for raw in setup.image_paths:
            path = raw if os.path.isabs(raw) else os.path.join(init_root or os.getcwd(), raw)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Context image not found: {path}")
            paths.append(path)
        cache_dir = os.environ.get("BO_EMBED_CACHE") or os.path.join(
            init_root or os.getcwd(), "ContextEmbeddingCache"
        )
        setup.embeddings = compute_image_embeddings(
            paths,
            model_name=setup.image_model,
            pretrained=setup.image_pretrained,
            cache_dir=cache_dir,
        )
    else:  # pragma: no cover - guarded by parse_context_config
        raise ValueError(f"Unknown embedding source '{setup.embedding_source}'")

    if setup.embeddings is not None:
        if setup.embeddings.shape[0] != setup.num_contexts:
            raise ValueError(
                f"Resolved {setup.embeddings.shape[0]} context embeddings for {setup.num_contexts} contexts."
            )
        if not np.all(np.isfinite(setup.embeddings)):
            raise ValueError("Resolved context embeddings contain NaN/Inf values.")
        if setup.normalize_embeddings:
            setup.embeddings = l2_normalize_rows(setup.embeddings)

    setup.embeddings_resolved = True
    return setup.embeddings


# -------------------- image embedding (optional dependency) --------------------
def _embed_images_with_open_clip(paths, model_name, pretrained):
    """Embed images with an open_clip vision tower (e.g. ViT-bigG-14 = ViT-G/14).

    Returns a (len(paths), m) float64 array. Raises a RuntimeError with install
    instructions when the optional dependencies are missing.
    """
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "embeddingSource=image requires the optional dependencies 'open_clip_torch' and 'pillow'. "
            "Install them into the Python environment used by the optimizer, e.g. "
            "`python -m pip install open_clip_torch pillow`. "
            f"Underlying import error: {e}"
        ) from e

    print(
        f"Loading image embedding model '{model_name}' (pretrained='{pretrained}') via open_clip. "
        "Large models such as ViT-bigG-14 download several GB of weights on first use.",
        flush=True,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for path in paths:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            features = model.encode_image(image)
            embeddings.append(features.squeeze(0).cpu().numpy().astype(np.float64))
    return np.stack(embeddings, axis=0)


# Indirection point so tests (or users) can plug in a different image embedder.
IMAGE_EMBEDDING_BACKEND = _embed_images_with_open_clip


def _image_cache_file(cache_dir, path, model_name, pretrained):
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(pretrained.encode("utf-8"))
    hasher.update(b"\x00")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_dir, f"{stem}.{hasher.hexdigest()[:20]}.npy")


def compute_image_embeddings(paths, model_name=DEFAULT_IMAGE_EMBEDDING_MODEL,
                             pretrained=DEFAULT_IMAGE_EMBEDDING_PRETRAINED,
                             cache_dir=None):
    """Compute (or load cached) image embeddings for the given files.

    Embeddings are cached as .npy files keyed by image content + model, so the
    (potentially very large) vision model only runs when an image or the model
    configuration changes.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    embeddings = [None] * len(paths)
    missing = []
    for i, path in enumerate(paths):
        if not cache_dir:
            missing.append(i)
            continue
        cache_file = _image_cache_file(cache_dir, path, model_name, pretrained)
        if os.path.exists(cache_file):
            try:
                cached = np.load(cache_file)
                if cached.ndim == 1 and np.all(np.isfinite(cached)):
                    embeddings[i] = np.asarray(cached, dtype=np.float64)
                    continue
                print(f"Warning: ignoring invalid embedding cache file: {cache_file}", flush=True)
            except (OSError, ValueError) as e:
                print(f"Warning: could not read embedding cache file {cache_file}: {e}", flush=True)
        missing.append(i)

    if missing:
        fresh = IMAGE_EMBEDDING_BACKEND([paths[i] for i in missing], model_name, pretrained)
        fresh = np.asarray(fresh, dtype=np.float64)
        if fresh.ndim != 2 or fresh.shape[0] != len(missing):
            raise ValueError(
                f"Image embedding backend returned shape {fresh.shape}, expected ({len(missing)}, m)."
            )
        for row, i in enumerate(missing):
            embeddings[i] = fresh[row]
            if cache_dir:
                cache_file = _image_cache_file(cache_dir, paths[i], model_name, pretrained)
                try:
                    np.save(cache_file, fresh[row])
                except OSError as e:
                    print(f"Warning: could not write embedding cache file {cache_file}: {e}", flush=True)

    dims = {e.shape[0] for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent image embedding dimensions: {sorted(dims)}")
    return np.stack(embeddings, axis=0)


# -------------------- warm-start context column --------------------
def context_indices_from_dataframe(x_df, setup, n_rows):
    """Map the optional 'Context' column of a warm-start parameter CSV to indices.

    Returns an int64 array of context indices (one per row). When the column is
    missing, every row is assigned to the current context (with a warning).
    """
    if setup is None:
        return None

    if CONTEXT_CSV_COLUMN not in x_df.columns:
        print(
            f"Warning: warm-start parameter CSV has no '{CONTEXT_CSV_COLUMN}' column; "
            f"assigning all rows to the current context '{setup.current_key}'.",
            flush=True,
        )
        return np.full(n_rows, setup.current_index, dtype=np.int64)

    key_by_lower = {k.lower(): i for i, k in enumerate(setup.keys)}
    indices = np.zeros(n_rows, dtype=np.int64)
    raw_values = list(x_df[CONTEXT_CSV_COLUMN])
    if len(raw_values) != n_rows:
        raise ValueError(
            f"Warm-start '{CONTEXT_CSV_COLUMN}' column has {len(raw_values)} rows, expected {n_rows}."
        )
    for i, raw in enumerate(raw_values):
        key = str(raw).strip()
        idx = key_by_lower.get(key.lower())
        if idx is None:
            raise ValueError(
                f"Warm-start row {i} references unknown context '{key}'. "
                f"Configured contexts: {setup.keys}"
            )
        indices[i] = idx
    return indices


# -------------------- task-column helpers (torch, lazy) --------------------
def append_task_column(x, task_indices):
    """Append a task/context-index column to a (n, d) tensor.

    ``task_indices`` may be a scalar (applied to every row) or a per-row
    sequence/array of length n. Returns a double tensor of shape (n, d+1).
    """
    import torch

    x_np = np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=np.float64)
    if x_np.ndim != 2:
        raise ValueError(f"Expected a 2D design matrix, got shape {x_np.shape}.")
    n_rows = x_np.shape[0]
    if np.isscalar(task_indices):
        col = np.full((n_rows, 1), float(task_indices), dtype=np.float64)
    else:
        col = np.asarray(task_indices, dtype=np.float64).reshape(-1, 1)
        if col.shape[0] != n_rows:
            raise ValueError(f"Task index count {col.shape[0]} != row count {n_rows}.")
    return torch.tensor(np.concatenate([x_np, col], axis=1), dtype=torch.double)


def strip_task_column(x):
    """Drop the trailing task/context column of a (n, d+1) tensor."""
    return x[..., :-1]


# -------------------- model construction (botorch, lazy) --------------------
def build_contextual_model(train_x_with_task, train_y, setup):
    """Build an LCE-M GP model (one LCEMGP per objective) and its MLL.

    The task feature is the last column of ``train_x_with_task``. Each LCEMGP is
    restricted to ``output_tasks=[current context]`` so its posterior over
    task-free inputs X (n x d) directly predicts outcomes for the current
    context — acquisition functions can then be used unchanged.
    """
    import torch
    from botorch.models import ModelListGP
    from botorch.models.contextual_multioutput import LCEMGP
    from gpytorch.constraints import Interval
    from gpytorch.kernels.rbf_kernel import RBFKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.mlls import SumMarginalLogLikelihood

    if not setup.embeddings_resolved:
        raise RuntimeError("resolve_embeddings() must be called before building the contextual model.")

    context_emb_feature = None
    if setup.embeddings is not None:
        context_emb_feature = torch.tensor(setup.embeddings, dtype=torch.double)

    # LCEMGP infers embedding rows from the observed task values by default;
    # pass explicit categorical indices so unobserved contexts stay addressable.
    context_cat_feature = torch.arange(setup.num_contexts, dtype=torch.double).unsqueeze(-1)
    all_tasks = list(range(setup.num_contexts))
    task_feature = train_x_with_task.shape[-1] - 1
    embs_dim_list = [1]  # one learned embedding dim for the single categorical feature

    models = []
    for j in range(train_y.shape[-1]):
        model_j = LCEMGP(
            train_X=train_x_with_task,
            train_Y=train_y[:, j: j + 1],
            task_feature=task_feature,
            context_cat_feature=context_cat_feature,
            context_emb_feature=context_emb_feature,
            embs_dim_list=embs_dim_list,
            output_tasks=[setup.current_index],
            all_tasks=all_tasks,
        )
        if context_emb_feature is not None:
            # BoTorch sizes the task kernel only for the learned embedding dims,
            # but evaluates it on learned + provided embeddings concatenated.
            # Rebuild it with the full dimensionality (same kernel/constraints).
            total_emb_dim = sum(embs_dim_list) + context_emb_feature.shape[-1]
            model_j.task_covar_module_base = RBFKernel(
                ard_num_dims=total_emb_dim,
                lengthscale_constraint=Interval(
                    0.0, 2.0, transform=None, initial_value=1.0
                ),
            ).to(train_x_with_task)
        models.append(model_j)

    if len(models) == 1:
        model = models[0]
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    else:
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def describe(setup):
    """One-line, log-friendly summary of the context configuration."""
    if setup is None:
        return "contextual optimization disabled"
    emb = "learned"
    if setup.embedding_source == "manual":
        emb = f"manual ({setup.embeddings.shape[1]}-dim)" if setup.embeddings is not None else "manual"
    elif setup.embedding_source == "image":
        emb = f"image via {setup.image_model}"
        if setup.embeddings is not None:
            emb += f" ({setup.embeddings.shape[1]}-dim)"
    return (
        f"{setup.num_contexts} context(s) {setup.keys}, current='{setup.current_key}' "
        f"(index {setup.current_index}), embeddings={emb}, normalize={setup.normalize_embeddings}"
    )
