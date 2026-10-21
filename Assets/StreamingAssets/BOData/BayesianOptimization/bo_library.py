# bo_library.py — the model library behind the BoTorch backends: BoTorch or botorch-rust.
#
# bo.py, mobo.py (plain and contextual mode) and MetaTAF (meta_mobo_runtime.py, meta_train.py
# and the openbo modules they drive) can run their GP models, acquisition functions and
# acquisition optimization on botorch-rust (https://github.com/M-Colley/botorch-rust, Python
# package `botorch_rs`), a Rust implementation of BoTorch with the same API paths. CABOP
# keeps its vendored optimizer; DBO uses botorch-rust with its GPyTorch temporal kernel.
#
# The environment variable BO_MODEL_LIBRARY selects it (Unity sets it from the inspector):
#   botorch      BoTorch (default): exactly the behaviour of the backends without this module.
#   botorch_rs   botorch-rust; an error at startup when it is not installed.
#   auto         botorch-rust when installed, otherwise BoTorch.
# A mode the installed botorch-rust cannot run (contextual optimization without its LCEMGP or
# ProvidedEmbeddingLCEMGP, MetaTAF without its PosteriorMeanModel) falls back to BoTorch with a
# console message, whatever was requested. The library actually used is printed in "Init OK"
# and recorded in the run folder (RunInfo.json, or MetaRunState.json for MetaTAF). It is a
# study factor: never switch it within a study.
#
# Kept identical across libraries on purpose:
#   * the initial Sobol design: botorch-rust's own draw_sobol_samples uses another scrambled
#     Sobol sequence, so the botorch_rs namespace draws the design with torch's SobolEngine
#     exactly as BoTorch does (README 8.14/8.15 promise the same design across backends);
#   * the BoTorch path itself: options, threads and names are what they were before.
#
# Standard library only at import, like context_support.py: torch, botorch and botorch_rs
# are imported where they are used, so this module imports in the stub-only CI environment.

import datetime
import hashlib
import importlib
import importlib.metadata
import os
import platform
import sys

ENV_VAR = "BO_MODEL_LIBRARY"
THREADS_ENV_VAR = "BO_TORCH_THREADS"

BOTORCH = "botorch"
BOTORCH_RS = "botorch_rs"
AUTO = "auto"
LIBRARIES = (BOTORCH, BOTORCH_RS)
REQUESTS = (BOTORCH, BOTORCH_RS, AUTO)
DEFAULT_REQUEST = BOTORCH
# Spellings accepted for botorch_rs: the project and distribution are called botorch-rust.
_ALIASES = {"botorch_rust": BOTORCH_RS, "botorch-rust": BOTORCH_RS, "botorch-rs": BOTORCH_RS}

# Distribution name of botorch_rs (pip install botorch-rust).
RS_DISTRIBUTION = "botorch-rust"

# Capabilities botorch-rust gains over time; a mode runs on botorch_rs only when its
# capability is importable. (module, attribute) pairs, BoTorch's API paths.
CONTEXTUAL_CAPABILITY = ("botorch_rs.models.contextual_multioutput", "LCEMGP")
CONTEXTUAL_PROVIDED_CAPABILITY = ("botorch_rs.models.contextual_multioutput", "ProvidedEmbeddingLCEMGP")
META_CAPABILITY = ("botorch_rs.models.deterministic", "PosteriorMeanModel")

# Contextual mode with provided (manual/image) embeddings uses this project's own LCE-M
# variant (context_support._provided_embedding_lcemgp_class: no learned embedding, a learned
# context-specific share), which BoTorch's LCEMGP signature cannot express. botorch_rs has it
# as ProvidedEmbeddingLCEMGP; a botorch_rs without it would fit a different model than the
# BoTorch path, so such runs then stay on BoTorch.
PROVIDED_EMBEDDINGS_REASON = (
    "contextual optimization with Manual or Image embeddings uses this project's "
    "provided-embedding LCE-M variant, which needs botorch_rs.models.contextual_multioutput."
    "ProvidedEmbeddingLCEMGP, which this botorch_rs lacks"
)

# botorch_rs's threads when BO_TORCH_THREADS is not set (bo.py sets that variable's default,
# 1, itself). Unity renders on the same machine while a suggestion is computed: a few threads
# give most of botorch_rs's multi-start speed-up without taking every core from the game.
DEFAULT_RS_THREADS_CAP = 4


class ModelLibraryError(RuntimeError):
    """The requested model library cannot be used (raised at backend startup)."""


# -------------------- selection --------------------
def normalize_request(value):
    """'botorch', 'botorch_rs' or 'auto' for a BO_MODEL_LIBRARY value (empty: the default)."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return DEFAULT_REQUEST
    text = _ALIASES.get(text, text)
    if text not in REQUESTS:
        raise ModelLibraryError(
            f"{ENV_VAR} must be one of {list(REQUESTS)} (or 'botorch-rust'), got {value!r}."
        )
    return text


def requested_library(environ=None):
    """The library requested through BO_MODEL_LIBRARY (normalized; ModelLibraryError if invalid)."""
    environ = os.environ if environ is None else environ
    return normalize_request(environ.get(ENV_VAR, ""))


def botorch_rs_status():
    """(installed, version, detail): whether botorch_rs imports, its version or why not."""
    try:
        import botorch_rs
    except ImportError as e:
        return False, None, f"botorch_rs is not installed ({e})"
    except Exception as e:  # a broken build (e.g. a DLL load failure) must not crash startup
        return False, None, f"botorch_rs failed to import ({type(e).__name__}: {e})"
    return True, str(getattr(botorch_rs, "__version__", "unknown")), ""


def has_capability(capability):
    """True when ``capability`` = (module, attribute) can be imported."""
    module_name, attribute = capability
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, attribute)


def contextual_capability(embedding_source):
    """(supported, reason): can botorch_rs run contextual mode with this embedding source?"""
    if embedding_source in ("manual", "image") and not has_capability(CONTEXTUAL_PROVIDED_CAPABILITY):
        return False, PROVIDED_EMBEDDINGS_REASON
    if not has_capability(CONTEXTUAL_CAPABILITY):
        module_name, attribute = CONTEXTUAL_CAPABILITY
        return False, f"contextual optimization needs {module_name}.{attribute}, which this botorch_rs lacks"
    return True, ""


def meta_capability():
    """(supported, reason): can MetaTAF (openbo) run on this botorch_rs?"""
    if not has_capability(META_CAPABILITY):
        module_name, attribute = META_CAPABILITY
        return False, f"MetaTAF needs {module_name}.{attribute}, which this botorch_rs lacks"
    return True, ""


def resolve(requested, contextual=None, meta=False):
    """Choose the model library: returns ``(library, reason)``.

    ``requested`` is a BO_MODEL_LIBRARY value. ``contextual`` is falsy for plain runs, else
    the context embedding source ('learned', 'manual', 'image'; True means 'learned').
    ``meta`` marks MetaTAF. botorch_rs requested but not installed raises ModelLibraryError;
    every other reason not to use botorch_rs falls back to BoTorch, and ``reason`` says why.
    """
    requested = normalize_request(requested)
    if requested == BOTORCH:
        return BOTORCH, f"{ENV_VAR}={requested}"
    installed, version, detail = botorch_rs_status()
    if not installed:
        if requested == BOTORCH_RS:
            raise ModelLibraryError(
                f"{ENV_VAR}=botorch_rs, but {detail}. Install botorch-rust into the optimizer's "
                "Python (README 8.16) or select BoTorch as the model library."
            )
        return BOTORCH, f"{ENV_VAR}=auto: {detail}; using BoTorch"
    if contextual:
        source = contextual if isinstance(contextual, str) else "learned"
        supported, why = contextual_capability(source)
        if not supported:
            return BOTORCH, f"{ENV_VAR}={requested}, botorch_rs {version} installed, but {why}; falling back to BoTorch"
    if meta:
        supported, why = meta_capability()
        if not supported:
            return BOTORCH, f"{ENV_VAR}={requested}, botorch_rs {version} installed, but {why}; falling back to BoTorch"
    return BOTORCH_RS, f"{ENV_VAR}={requested}, botorch_rs {version}"


def is_fallback(requested, library):
    """True when botorch_rs was asked for (botorch_rs or auto) but BoTorch is used."""
    return normalize_request(requested) != BOTORCH and library == BOTORCH


def fallback_message(requested, library, reason):
    """Console warning for a run that asked for botorch_rs but runs on BoTorch, else None.

    'auto' without botorch_rs installed is no warning: that is what auto means (the reason
    line printed with the library says it).
    """
    if not is_fallback(requested, library):
        return None
    if normalize_request(requested) == AUTO and not botorch_rs_status()[0]:
        return None
    return f"Warning: the model library falls back to BoTorch: {reason}"


# -------------------- names --------------------
# Every name bo.py, mobo.py and context_support bind from the model library: name ->
# (BoTorch module, botorch_rs module). GPyTorch's MLLs live in botorch_rs.mlls.
_NAME_MODULES = {
    "qLogNoisyExpectedImprovement": ("botorch.acquisition.logei", "botorch_rs.acquisition.logei"),
    "qLogNoisyExpectedHypervolumeImprovement": (
        "botorch.acquisition.multi_objective.logei",
        "botorch_rs.acquisition.multi_objective.logei",
    ),
    "fit_gpytorch_mll": ("botorch.fit", "botorch_rs.fit"),
    "optimize_acqf": ("botorch.optim.optimize", "botorch_rs.optim.optimize"),
    "SobolQMCNormalSampler": ("botorch.sampling.normal", "botorch_rs.sampling.normal"),
    "draw_sobol_samples": ("botorch.utils.sampling", None),  # botorch_rs: draw_sobol_samples below
    "SingleTaskGP": ("botorch.models", "botorch_rs.models"),
    "ModelListGP": ("botorch.models", "botorch_rs.models"),
    "LCEMGP": ("botorch.models.contextual_multioutput", "botorch_rs.models.contextual_multioutput"),
    # BoTorch: context_support._provided_embedding_lcemgp_class(), this project's own subclass
    "ProvidedEmbeddingLCEMGP": (None, "botorch_rs.models.contextual_multioutput"),
    "ExactMarginalLogLikelihood": ("gpytorch.mlls", "botorch_rs.mlls"),
    "SumMarginalLogLikelihood": ("gpytorch.mlls", "botorch_rs.mlls"),
}


class _Namespace:
    """The names of one model library, imported on first access (``namespace(library).X``)."""

    def __init__(self, library):
        if library not in LIBRARIES:
            raise ValueError(f"Unknown model library {library!r}; expected one of {list(LIBRARIES)}.")
        self.library = library

    def __getattr__(self, name):
        if name not in _NAME_MODULES:
            raise AttributeError(f"{name!r} is not a model-library name of the BoTorch backends")
        if self.library == BOTORCH_RS and name == "draw_sobol_samples":
            value = draw_sobol_samples
        elif self.library == BOTORCH_RS and name == "SobolQMCNormalSampler":
            value = _rs_sobol_sampler_factory()
        else:
            module_name = _NAME_MODULES[name][0 if self.library == BOTORCH else 1]
            if module_name is None:
                raise AttributeError(f"{name!r} has no {self.library} counterpart in this namespace")
            value = getattr(importlib.import_module(module_name), name)
        setattr(self, name, value)  # cache: __getattr__ only runs for missing attributes
        return value


def namespace(library):
    """Lazy namespace of the names the backends bind for ``library`` (see _NAME_MODULES)."""
    return _Namespace(library)


def _rs_sobol_sampler_factory():
    """botorch_rs's SobolQMCNormalSampler, accepting every seed BoTorch accepts.

    botorch_rs takes unsigned 64-bit seeds; Unity's Seed may be negative (BoTorch's sampler
    accepts it). Seeds are mapped with to_rs_seed, so the same Unity seed always gives the
    same base samples.
    """
    from botorch_rs.sampling.normal import SobolQMCNormalSampler

    def make_sampler(sample_shape, seed=None, **kwargs):
        return SobolQMCNormalSampler(sample_shape, seed=None if seed is None else to_rs_seed(seed), **kwargs)

    return make_sampler


def to_rs_seed(seed):
    """A seed as botorch_rs's unsigned 64-bit seed: wrapped modulo 2**64, so -1 -> 2**64 - 1."""
    return int(seed) % (1 << 64)


# -------------------- initial design --------------------
def draw_sobol_samples(bounds, n, q, seed=None):
    """BoTorch's ``draw_sobol_samples(bounds, n, q, seed=seed)`` (no batch_shape), bit for bit.

    torch's SobolEngine with Owen scrambling, exactly as botorch.utils.sampling draws it, so
    the initial design is identical whichever model library runs (botorch_rs's own
    draw_sobol_samples uses another Sobol sequence). Returns an ``n x q x d`` tensor.
    """
    from torch.quasirandom import SobolEngine

    d = bounds.shape[-1]
    sobol_engine = SobolEngine(q * d, scramble=True, seed=seed)
    samples_raw = sobol_engine.draw(n, dtype=bounds.dtype)
    samples_raw = samples_raw.view(n, q, d).to(device=bounds.device)
    return samples_raw * (bounds[1] - bounds[0]) + bounds[0]


# -------------------- acquisition optimization --------------------
def acqf_options(library, base, seed, iteration):
    """``optimize_acqf`` options: ``base`` plus, on botorch_rs only, a per-iteration seed.

    botorch_rs ignores torch.manual_seed and draws its raw samples and initial conditions from
    its own generator; ``seed = SEED + iteration`` makes every suggestion reproducible and
    different from the previous one (as BoTorch's global torch generator does after
    torch.manual_seed). The BoTorch options stay exactly ``base``.
    """
    options = dict(base)
    if library == BOTORCH_RS:
        options["seed"] = to_rs_seed(int(seed) + int(iteration))
    return options


# -------------------- threads --------------------
def threads_from_env(environ=None):
    """BO_TORCH_THREADS as an int (>= 1), or None when unset; ValueError when invalid."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(THREADS_ENV_VAR, "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{THREADS_ENV_VAR} must be an integer >= 1, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{THREADS_ENV_VAR} must be >= 1, got {value}")
    return value


def default_rs_threads(cpu_count=None):
    """botorch_rs's threads when BO_TORCH_THREADS is unset: half the logical CPUs, at most 4."""
    cpu_count = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(DEFAULT_RS_THREADS_CAP, (cpu_count or 2) // 2))


def configure_rs_threads(library, threads=None):
    """Set botorch_rs's thread count (``threads`` or default_rs_threads()); returns it.

    Does nothing (returns None) for BoTorch, whose torch threads the backends set as before.
    """
    if library != BOTORCH_RS:
        return None
    import botorch_rs

    botorch_rs.set_num_threads(int(threads) if threads is not None else default_rs_threads())
    return int(botorch_rs.get_num_threads())


# -------------------- provenance --------------------
# Distributions whose versions decide the numbers a run produces.
VERSION_DISTRIBUTIONS = (
    "torch", "botorch", "gpytorch", "linear_operator", RS_DISTRIBUTION,
    "numpy", "scipy", "pandas", "moocore",
)


def distribution_versions(distributions=VERSION_DISTRIBUTIONS):
    """Installed versions of ``distributions`` (None = absent)."""
    versions = {}
    for dist in distributions:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def botorch_rs_record():
    """Identity of the imported botorch_rs: version, location and SHA-256 of its extension.

    The extension's hash distinguishes source builds and identifies the wheel's native
    payload (like MetaRunState.json's openbo module hashes).
    """
    module = sys.modules.get("botorch_rs")
    if module is None:
        return None
    record = {"version": str(getattr(module, "__version__", "unknown")),
              "path": os.path.dirname(getattr(module, "__file__", "") or ""),
              "extension_sha256": None}
    extension = sys.modules.get("botorch_rs._botorch_rs")
    extension_file = getattr(extension, "__file__", None)
    if extension_file and os.path.isfile(extension_file):
        try:
            record["extension_sha256"] = _sha256_file(extension_file)
        except OSError:
            pass
    return record


def library_record(requested, library, reason, threads=None):
    """The model-library block of RunInfo.json / MetaRunState.json."""
    record = {
        "requested": normalize_request(requested),
        "library": library,
        "reason": reason,
        "fallback": is_fallback(requested, library),
        "version": None,
        "threads": threads,
        "botorch_rs": None,
    }
    if library == BOTORCH_RS:
        record["botorch_rs"] = botorch_rs_record()
        record["version"] = (record["botorch_rs"] or {}).get("version")
    else:
        record["version"] = distribution_versions((BOTORCH,))[BOTORCH]
    return record


def describe(library_info):
    """'botorch 0.18.1' / 'botorch_rs 0.1.0-dev, 4 threads' for the console."""
    if not library_info:
        return BOTORCH
    text = f"{library_info['library']} {library_info.get('version') or ''}".strip()
    if library_info.get("threads"):
        text += f", {library_info['threads']} thread{'s' if library_info['threads'] != 1 else ''}"
    return text


RUN_INFO_FILENAME = "RunInfo.json"
RUN_INFO_SCHEMA_VERSION = 1


def run_info(generator, seed, library_info):
    """Content of RunInfo.json: what produced a bo.py / mobo.py run."""
    return {
        "schema_version": RUN_INFO_SCHEMA_VERSION,
        "generator": generator,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "model_library": library_info,
        "library_versions": distribution_versions(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def bound_library(objects):
    """The library ('botorch' / 'botorch_rs') the given functions or classes come from, or None.

    Used to confirm which library a third-party module (openbo) really bound: None when no
    object reveals it, ValueError-free.
    """
    found = set()
    for obj in objects:
        module_name = getattr(obj, "__module__", None) or ""
        root = module_name.split(".", 1)[0]
        if root == "botorch_rs":
            found.add(BOTORCH_RS)
        elif root in ("botorch", "gpytorch"):
            found.add(BOTORCH)
    if len(found) == 1:
        return found.pop()
    return None


# openbo modules whose module-level names show which library openbo bound after reading
# BO_MODEL_LIBRARY; an explicit MODEL_LIBRARY attribute on them takes precedence.
OPENBO_LIBRARY_MODULES = ("openbo.optimizers.mobo_botorch", "openbo.optimizers.mobo_taf",
                          "openbo.acquisition.taf_mo_ehvi")
_OPENBO_LIBRARY_NAMES = ("optimize_acqf", "fit_gpytorch_mll", "SingleTaskGP", "ModelListGP")


def openbo_library():
    """The model library the imported openbo uses, or None when it cannot be determined."""
    objects = []
    for module_name in OPENBO_LIBRARY_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        explicit = getattr(module, "MODEL_LIBRARY", None)
        if isinstance(explicit, str):
            try:
                normalized = normalize_request(explicit)
            except ModelLibraryError:
                normalized = None
            if normalized in LIBRARIES:
                return normalized
        objects.extend(getattr(module, name) for name in _OPENBO_LIBRARY_NAMES if hasattr(module, name))
    return bound_library(objects)
