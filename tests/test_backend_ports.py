"""Two real loopback backend protocol processes must keep their studies separate."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
SERVER = """
import socket, sys
sys.path.insert(0, sys.argv[1])
import bo_protocol as p
with socket.socket() as listener:
    p.configure_listener_socket(listener)
    connection = p.accept_unity_connection(listener, 10)
    try:
        init = p.receive_init_message(connection, 10)
        p.send_json_line(connection, {"port": p.PORT, "token": init["token"]})
    finally:
        p.close_connection(connection)
"""


class BackendPortIsolationTests(unittest.TestCase):
    def test_two_simultaneous_backend_processes_receive_only_their_own_study(self):
        probes = [socket.socket(), socket.socket()]
        try:
            for probe in probes:
                probe.bind(("127.0.0.1", 0))
            ports = [probe.getsockname()[1] for probe in probes]
        finally:
            for probe in probes:
                probe.close()
        self.assertEqual(len(set(ports)), 2)
        processes = []
        connections = []
        try:
            for port in ports:
                env = dict(os.environ, BO_PORT=str(port), PYTHONDONTWRITEBYTECODE="1")
                processes.append(subprocess.Popen(
                    [sys.executable, "-c", SERVER, str(BACKEND)], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                ))
            for process, port in zip(processes, ports):
                deadline = time.monotonic() + 10
                while True:
                    try:
                        connection = socket.create_connection(("127.0.0.1", port), timeout=1)
                        connection.settimeout(10)
                        connections.append(connection)
                        break
                    except OSError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail(f"Backend did not listen on its port {port}")
                        time.sleep(0.02)
            for index, connection in enumerate(connections):
                connection.sendall((json.dumps({"type": "init", "token": f"study-{index}"}) + "\n").encode())
            for index, (connection, port) in enumerate(zip(connections, ports)):
                with connection.makefile("rb") as reader:
                    self.assertEqual(json.loads(reader.readline()), {"port": port, "token": f"study-{index}"})
            for process in processes:
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, output)
        finally:
            for connection in connections:
                connection.close()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
