# Python integration (`interactive-bo`)

The Python client for the [iBO protocol](../protocol/README.md) and a launcher for the iBO server
(`Backend/ibo_server.py`). Any program that can show a design to a person and measure the
response can run a study with it, without Unity.

```python
from interactive_bo import IboClient, Objective, Parameter, Study, launch_server

study = Study(
    parameters=[Parameter("size", 0.2, 2.0), Parameter("contrast", 0.0, 1.0)],
    objectives=[Objective("usability", 0, 100), Objective("time", 0, 10, minimize=True)],
    sampling_iterations=4, optimization_iterations=10, user_id="P01", condition_id="A",
)

with launch_server(python="/path/to/python-with-botorch") as server:
    with IboClient(port=server.port) as client:
        client.start(study)
        while (design := client.next_design()) is not None:
            client.report(run_trial_with_participant(design.values))  # {"usability": ..., "time": ...}
```

- `Study` builds the init message. `config={...}` passes further settings (backend options, warm start),
  `extra={...}` further init fields (`context`, `cabopGroupCosts`), as listed in the protocol.
- `run_study(study, evaluate, port=...)` runs the whole loop with a function.
- The run logs land in `<log_root>/<userId>/<conditionId>/run/` (default `log_root`: `./LogData`),
  in the same format as the Unity integration.

The server needs Python 3.13 with the packages of
[Backend/requirements.txt](../Backend/requirements.txt). The client needs nothing.

```bash
python -m interactive_bo demo --python /path/to/python-with-botorch
```

```bash
python -m interactive_bo serve --python /path/to/python-with-botorch --port 56001
```

Tests (the end-to-end test runs when `IBO_TEST_PYTHON` points to an interpreter with the optimizer stack):

```bash
python -m unittest discover Python/tests
```
