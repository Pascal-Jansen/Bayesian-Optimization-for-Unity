# botorch_common.py — algorithm-side helpers shared by the BoTorch backends bo.py and mobo.py.
#
# The two backends differ in their acquisition function, their run metric (best objective vs
# hypervolume) and their observation flag (IsBest vs IsPareto). Warm-start loading, model
# construction, the context bookkeeping and the observation-log append were line-for-line
# copies; they live here once. Each backend keeps thin wrappers that pass its module state, so
# its call sites (and tests that set e.g. bo.CSV_PATH_PARAMETERS) stay unchanged.
#
# Only numpy and the torch-free sibling modules are imported at module scope. torch, botorch,
# gpytorch and pandas are imported where they are used, like in context_support.py: the module
# then imports without torch (CI) and always uses the torch in sys.modules at call time.

import os
import time

import numpy as np

import bo_normalize
import context_support
from bo_protocol import log_exists, read_observation_log, write_dataframe_csv


# -------------------- context bookkeeping --------------------
def fixed_observation_columns(context_setup, flag_column):
    """Fixed leading columns of ObservationsPerEvaluation.csv (objectives and parameters follow)."""
    cols = ['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase']
    if context_setup is not None:
        cols.append(context_support.CONTEXT_CSV_COLUMN)
    return cols + [flag_column]


def observation_context_cells(context_setup):
    """Extra CSV cells for the context column (empty when contexts are disabled)."""
    return [context_setup.current_key] if context_setup is not None else []


def current_context_mask(x_sample, context_setup):
    """Boolean numpy row mask of observations that belong to the current context.

    In contextual mode the last column of ``x_sample`` carries the context/task
    index; only current-context rows should count towards run metrics.
    """
    n_rows = int(x_sample.shape[0])
    if context_setup is None:
        return np.ones(n_rows, dtype=bool)
    x_np = x_sample.cpu().numpy() if hasattr(x_sample, "cpu") else np.asarray(x_sample)
    return np.asarray(x_np)[:, -1] == float(context_setup.current_index)


def acquisition_baseline(train_x, context_setup):
    """Baseline design points for the acquisition function.

    In contextual mode the trailing task column is stripped: the LCE-M GP is
    restricted to the current context, so its posterior over plain (n x d)
    design points already predicts current-context outcomes.
    """
    if context_setup is not None:
        return context_support.strip_task_column(train_x)
    return train_x


# -------------------- model --------------------
def initialize_model(train_x, train_obj, context_setup):
    """(mll, model): an LCE-M GP in contextual mode, otherwise a SingleTaskGP."""
    if context_setup is not None:
        return context_support.build_contextual_model(train_x, train_obj, context_setup)
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# -------------------- warm start --------------------
def load_warm_start_data(parameters_csv, objectives_csv, parameter_names, parameters_info,
                         objective_names, objectives_info, context_setup,
                         normalize_param_column, normalize_obj_column):
    """Read the warm-start CSVs into (train_x, train_y) in the canonical frame.

    Paths are relative to $BO_INIT_ROOT (default ./InitData). ``normalize_param_column(col,
    lo, hi)`` and ``normalize_obj_column(col, lo, hi, minflag)`` are the backend's column
    transforms (the objective one carries the backend's warm-start objective format). In
    contextual mode the optional 'Context' column assigns each row to a context and becomes
    the trailing task column of train_x.
    """
    import pandas as pd
    import torch

    if not parameters_csv or not objectives_csv:
        raise ValueError("Warm start is enabled, but initial CSV paths are missing.")

    init_root = os.environ.get("BO_INIT_ROOT") or os.path.join(os.getcwd(), "InitData")
    x_path = os.path.join(init_root, parameters_csv)
    y_path = os.path.join(init_root, objectives_csv)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Warm-start parameter CSV not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Warm-start objective CSV not found: {y_path}")

    x_df = pd.read_csv(x_path, delimiter=';', encoding='utf-8',
                       dtype={context_support.CONTEXT_CSV_COLUMN: str})
    y_df = pd.read_csv(y_path, delimiter=';', encoding='utf-8')

    missing_param_cols = [k for k in parameter_names if k not in x_df.columns]
    missing_obj_cols = [k for k in objective_names if k not in y_df.columns]
    if missing_param_cols:
        raise ValueError(f"Warm-start parameter CSV is missing columns: {missing_param_cols}")
    if missing_obj_cols:
        raise ValueError(f"Warm-start objective CSV is missing columns: {missing_obj_cols}")

    x_raw = x_df[parameter_names].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    y_raw = y_df[objective_names].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if x_raw.shape[0] != y_raw.shape[0]:
        raise ValueError(f"Warm-start rows mismatch: parameters={x_raw.shape[0]}, objectives={y_raw.shape[0]}")
    if x_raw.shape[0] < 1:
        raise ValueError("Warm-start CSVs must contain at least one data row.")
    if not np.all(np.isfinite(x_raw)):
        raise ValueError("Warm-start parameter CSV contains NaN/Inf values.")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("Warm-start objective CSV contains NaN/Inf values.")

    x_norm = np.zeros_like(x_raw, dtype=np.float64)
    for j in range(len(parameter_names)):
        lo, hi = parameters_info[j]
        x_norm[:, j] = normalize_param_column(x_raw[:, j], lo, hi)

    y_norm = np.zeros_like(y_raw, dtype=np.float64)
    for j in range(len(objective_names)):
        lo, hi, minflag = objectives_info[j]
        y_norm[:, j] = normalize_obj_column(y_raw[:, j], lo, hi, minflag)

    if not np.all(np.isfinite(x_norm)):
        raise ValueError("Warm-start normalized parameters contain non-finite values.")
    if not np.all(np.isfinite(y_norm)):
        raise ValueError("Warm-start normalized objectives contain non-finite values.")

    train_x = torch.tensor(x_norm, dtype=torch.double)
    if context_setup is not None:
        # Optional 'Context' column assigns each warm-start row to a context.
        ctx_indices = context_support.context_indices_from_dataframe(
            x_df, context_setup, x_norm.shape[0]
        )
        train_x = context_support.append_task_column(train_x, ctx_indices)
    return train_x, torch.tensor(y_norm, dtype=torch.double)


# -------------------- observation log --------------------
def append_optimization_observation(obs_csv, x_sample, y_sample, *, context_setup,
                                    parameters_info, objectives_info, id_cells,
                                    expected_columns, flag_column, current_context_flags):
    """Log the newest row of (x_sample, y_sample) to ObservationsPerEvaluation.csv.

    The row is denormalized to original units and logged under the run's Iteration: the
    number of current-context rows (warm-start rows from *other* contexts inform the model but
    are not iterations of this run). ``current_context_flags(y_sample, ctx_mask)`` returns the
    'TRUE'/'FALSE' flags of the current-context rows; they replace ``flag_column`` of the log's
    tail, so older, unrelated rows keep theirs. Returns the logged Iteration.
    """
    import pandas as pd

    x_np = x_sample.clone().cpu().numpy()
    y_np = y_sample.clone().cpu().numpy()
    ctx_mask = current_context_mask(x_sample, context_setup)
    iteration_index = int(np.sum(ctx_mask))
    problem_dim = len(parameters_info)
    if context_setup is not None:
        # Drop the trailing context/task column for parameter logging.
        x_np = np.asarray(x_np)[:, :problem_dim]

    # denormalize the last row
    for j in range(problem_dim):
        lo, hi = parameters_info[j]
        x_np[-1][j] = bo_normalize.denormalize_to_original_param(x_np[-1][j], lo, hi)
    for j in range(len(objectives_info)):
        lo, hi, minflag = objectives_info[j]
        y_np[-1][j] = bo_normalize.denormalize_to_original_obj(y_np[-1][j], lo, hi, minflag)

    if log_exists(obs_csv):
        df = read_observation_log(obs_csv)
        if list(df.columns) != expected_columns:
            raise ValueError(
                f"ObservationsPerEvaluation.csv columns mismatch. "
                f"Expected {expected_columns}, got {list(df.columns)}"
            )
    else:
        df = pd.DataFrame(columns=expected_columns)

    new_row = pd.DataFrame([[*id_cells,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                             iteration_index, 'optimization', *observation_context_cells(context_setup),
                             'FALSE', *y_np[-1], *x_np[-1]]], columns=df.columns)
    if df.empty:
        df = new_row.copy()
    else:
        df = pd.concat([df, new_row], ignore_index=True)

    flags = current_context_flags(y_sample, ctx_mask)
    df[flag_column] = df[flag_column].astype(str)
    if len(flags) >= len(df):
        df[flag_column] = flags[-len(df):]
    elif len(flags) > 0:
        tail = df.index[-len(flags):]
        df.loc[tail, flag_column] = flags

    write_dataframe_csv(obs_csv, df)
    return iteration_index
