"""Command line: ``python -m interactive_bo serve`` or ``python -m interactive_bo demo``."""

from __future__ import annotations

import argparse
import sys

from .server import launch_server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ibo", description="Interactive Bayesian Optimization (iBO)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--python", help="interpreter with the optimizer stack (default: this one)")
        p.add_argument("--log-root", help="folder for the run logs (default: ./LogData)")
        p.add_argument("--backend-dir", help="folder holding ibo_server.py (default: IBO_BACKEND_DIR or this checkout)")

    serve = sub.add_parser("serve", help="run the iBO server for one study; clients connect to 127.0.0.1:PORT")
    common(serve)
    serve.add_argument("--port", type=int, default=56001, help="TCP port (default: %(default)s, as Unity)")
    serve.add_argument("--init-root", help="warm-start and context data (default: the backend's InitData)")

    demo = sub.add_parser("demo", help="run a short study with a simulated participant")
    common(demo)
    demo.add_argument("--sampling", type=int, default=4, help="sampling iterations (default: %(default)s)")
    demo.add_argument("--iterations", type=int, default=4, help="optimization iterations (default: %(default)s)")

    args = parser.parse_args(argv)
    if args.command == "serve":
        server = launch_server(python=args.python, port=args.port, log_root=args.log_root,
                               init_root=args.init_root, backend_dir=args.backend_dir, echo=True)
        print(f"iBO server listening on 127.0.0.1:{server.port}; logs in {server.log_root}", flush=True)
        try:
            return server.wait()
        except KeyboardInterrupt:
            server.stop()
            return 130

    from .demo import run_demo  # noqa: E402 (only the demo needs it)
    history = run_demo(python=args.python, log_root=args.log_root, sampling_iterations=args.sampling,
                       optimization_iterations=args.iterations)
    for row in history:
        params = ", ".join(f"{k}={v:.3f}" for k, v in row["parameters"].items())
        objs = ", ".join(f"{k}={v:.2f}" for k, v in row["objectives"].items())
        print(f"iteration {row['iteration']}: {params} -> {objs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
