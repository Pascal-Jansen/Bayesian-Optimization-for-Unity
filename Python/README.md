# iBO for Python (`interactive-bo`)

The Python client for the [iBO protocol](../protocol/README.md) and a launcher for the iBO server
(`Backend/ibo_server.py`). Any program that can show a design to a person and measure the response can run a study
with it, without a game engine: experiment scripts, PsychoPy, robot or lab-hardware controllers.

## Install

From the repository root (the launcher finds `Backend/` there):

```bash
python -m pip install ./Python
```

For development, install it editable (`python -m pip install -e ./Python`). The client needs only the standard library
and runs on Python 3.9 or newer. The server needs **Python 3.13** with the packages of
[Backend/requirements.txt](../Backend/requirements.txt); pass that interpreter as `python` (`--python`).

## Use

```python
from interactive_bo import IboClient, Objective, Parameter, Study, launch_server

study = Study(
    parameters=[Parameter("size", 0.2, 2.0), Parameter("contrast", 0.0, 1.0)],
    objectives=[Objective("usability", 0, 100), Objective("time", 0, 10, minimize=True)],
    sampling_iterations=4, optimization_iterations=10, user_id="P01", condition_id="A",
)

with launch_server(python="/path/to/python-with-the-requirements") as server:
    with IboClient(port=server.port) as client:
        client.start(study)
        while (design := client.next_design()) is not None:
            client.report(run_trial_with_participant(design.values))  # {"usability": ..., "time": ...}
```

- `Study` builds the init message. `backend=` selects the optimizer (`botorch`, `cabop`, `dbo`, `meta-taf`);
  `config={...}` passes further settings (backend options, warm start) and `extra={...}` further init fields
  (`context`, `cabopGroupCosts`), as listed in the protocol.
- `run_study(study, evaluate, port=...)` runs the whole loop with a function.
- The run logs land in `<log_root>/<userId>/<conditionId>/run/` (default `log_root`: `./LogData`), in the same format
  as the Unity integration.
- `launch_server` looks for the backend in `IBO_BACKEND_DIR`, then in the repository the package lives in, then in
  `./Backend`; pass `backend_dir=` to use another copy.

## Command line

Run a complete study with a simulated participant:

```bash
ibo demo --python /path/to/python-with-the-requirements
```

Run the server for another program to connect to (one study per start):

```bash
ibo serve --python /path/to/python-with-the-requirements --port 56001
```

`python -m interactive_bo` works the same way without the `ibo` command.

## Tests

The end-to-end tests run real studies when `IBO_TEST_PYTHON` points to an interpreter with the requirements:

```bash
python -m unittest discover Python/tests
```
