"""Run the iBO studies that the README animations show and store them in studies.json.

Usage: python tools/readme_figures/run_studies.py --python <interpreter with Backend/requirements.txt>
DBO needs botorch-rust in that interpreter. make_animations.py draws the GIFs from studies.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRIFT_EVALS = 24


def preference(x):
    """One parameter: a lower peak near 0.22 and the best design near 0.71."""
    return 1.0 + 4.5 * math.exp(-((x - 0.22) / 0.09) ** 2) + 7.5 * math.exp(-((x - 0.71) / 0.11) ** 2)


def comfort(x1, x2):
    return 10.0 - 8.0 * ((x1 - 0.2) ** 2 + (x2 - 0.3) ** 2)


def speed(x1, x2):
    return 10.0 - 8.0 * ((x1 - 0.8) ** 2 + (x2 - 0.7) ** 2)


def drifting_peak(t):
    """The best design moves from 0.25 to 0.75 while the study runs (the participant adapts)."""
    return 0.25 + 0.5 * min(1.0, t / (DRIFT_EVALS - 1))


def drifting(x, t):
    return 1.0 + 9.0 * math.exp(-((x - drifting_peak(t)) / 0.12) ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="interpreter with the optimizer stack")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "Python"))
    from interactive_bo import Objective, Parameter, Study, launch_server, run_study

    def run(name, study, evaluate):
        with launch_server(python=args.python, backend_dir=ROOT / "Backend", log_root=HERE / "runs" / name) as server:
            history = run_study(study, evaluate, port=server.port, timeout=600)
        print(name, len(history), "evaluations", flush=True)
        return history

    results = {"bo1d": run("bo1d", Study(
        parameters=[Parameter("x", 0.0, 1.0)], objectives=[Objective("rating", 0.0, 10.0)],
        sampling_iterations=3, optimization_iterations=9), lambda d: {"rating": preference(d["x"])})}
    results["mobo"] = run("mobo", Study(
        parameters=[Parameter("x1", 0.0, 1.0), Parameter("x2", 0.0, 1.0)],
        objectives=[Objective("comfort", 0.0, 10.0), Objective("speed", 0.0, 10.0)],
        sampling_iterations=6, optimization_iterations=14),
        lambda d: {"comfort": comfort(d["x1"], d["x2"]), "speed": speed(d["x1"], d["x2"])})
    for backend in ("dbo", "botorch"):
        ratings = []

        def evaluate(d, ratings=ratings):
            ratings.append(drifting(d["x"], len(ratings)))
            return {"rating": ratings[-1]}

        results["drift_" + backend] = run("drift_" + backend, Study(
            parameters=[Parameter("x", 0.0, 1.0)], objectives=[Objective("rating", 0.0, 10.0)], backend=backend,
            sampling_iterations=4, optimization_iterations=DRIFT_EVALS - 4), evaluate)
    (HERE / "studies.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
