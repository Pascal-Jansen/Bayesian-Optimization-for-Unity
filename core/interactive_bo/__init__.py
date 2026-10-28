"""Interactive Bayesian Optimization (IBO): human-in-the-loop Bayesian optimization for any engine.

The IBO server proposes designs and learns from the people who evaluate them; integrations (Unity,
Unreal, web, Python) show the designs and report the results over the IBO protocol
(protocol/README.md). This package holds the Python client and the server launcher.
"""

from .client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    Design,
    IboClient,
    Objective,
    Parameter,
    ProtocolError,
    Study,
    run_study,
)
from .server import ServerProcess, default_backend_dir, launch_server

__version__ = "0.1.0.dev0"

__all__ = [
    "DEFAULT_HOST", "DEFAULT_PORT", "PROTOCOL_VERSION", "Design", "IboClient", "Objective", "Parameter",
    "ProtocolError", "ServerProcess", "Study", "default_backend_dir", "launch_server", "run_study",
]
