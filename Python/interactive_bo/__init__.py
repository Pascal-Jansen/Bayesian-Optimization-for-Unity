"""Interactive Bayesian Optimization (iBO): human-in-the-loop Bayesian optimization for any engine.

The iBO server proposes designs and learns from the people who evaluate them; integrations (Unity,
Unreal, web, Python) show the designs and report the results over the iBO protocol
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

__version__ = "2.0.0rc1"

__all__ = [
    "DEFAULT_HOST", "DEFAULT_PORT", "PROTOCOL_VERSION", "Design", "IboClient", "Objective", "Parameter",
    "ProtocolError", "ServerProcess", "Study", "default_backend_dir", "launch_server", "run_study",
]
