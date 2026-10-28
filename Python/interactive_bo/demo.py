"""A complete iBO study without any engine: a simulated participant rates button designs.

The participant's usability rating peaks at a medium size and high contrast, while larger buttons
are always faster to hit, so the two objectives conflict and the optimizer has to find the trade-off.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from .client import Objective, Parameter, Study, run_study
from .server import launch_server

STUDY_PARAMETERS = [Parameter("size", 0.2, 2.0), Parameter("contrast", 0.0, 1.0)]
STUDY_OBJECTIVES = [Objective("usability", 0.0, 100.0), Objective("time", 0.0, 10.0, minimize=True)]


def simulated_participant(rng: random.Random):
    def evaluate(design: Dict[str, float]) -> Dict[str, float]:
        size, contrast = design["size"], design["contrast"]
        usability = 100.0 * math.exp(-((size - 1.1) ** 2) / 0.3 - ((contrast - 0.8) ** 2) / 0.15)
        time = 1.0 + 2.5 / size
        return {
            "usability": min(100.0, max(0.0, usability + rng.gauss(0.0, 3.0))),
            "time": min(10.0, max(0.0, time + rng.gauss(0.0, 0.2))),
        }
    return evaluate


def run_demo(python: Optional[str] = None, log_root: Optional[str] = None, sampling_iterations: int = 4,
             optimization_iterations: int = 4, seed: int = 3, echo: bool = False) -> List[dict]:
    study = Study(STUDY_PARAMETERS, STUDY_OBJECTIVES, sampling_iterations=sampling_iterations,
                  optimization_iterations=optimization_iterations, seed=seed,
                  user_id="demo", condition_id="simulated")
    with launch_server(python=python, log_root=log_root, echo=echo) as server:
        history = run_study(study, simulated_participant(random.Random(seed)), port=server.port)
        server.wait(timeout=60)
    return history
