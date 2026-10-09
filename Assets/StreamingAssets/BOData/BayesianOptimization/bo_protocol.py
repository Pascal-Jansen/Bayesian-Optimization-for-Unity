# bo_protocol.py — Unity <-> Python plumbing shared by every BOforUnity backend.
#
# bo.py, mobo.py, dbo_runtime.py, cabop_runtime.py and meta_mobo_runtime.py speak one wire
# protocol (Python is a TCP server on 127.0.0.1:56001; newline-delimited UTF-8 JSON: Unity
# sends one init message, then Python drives a parameters -> objectives loop) and write the
# same CSV family. That plumbing lives here once, so a protocol or robustness fix reaches every
# backend: the listener and the NDJSON reader, init-message parsing and validation helpers,
# run-folder naming and the lock-tolerant log writer.
#
# Standard library only (pandas is imported lazily by read_observation_log), so the module
# imports in any environment. Backends import the names they use (`from bo_protocol import
# send_json_line, ...`): the names stay module attributes of each backend, and monkeypatching
# e.g. bo.send_json_line still affects bo's own calls.
#
# Unity -> Python message types: init, objectives, stop (the study ends on purpose: a
# perfect-rating stop while Python waits for objectives, or an invalid configuration in place of
# the init). Unknown types are ignored, so Unity can add status messages.
# Python -> Unity: parameters (with the Iteration the design is logged under), coverage,
# tempCoverage, optimization_finished.

import collections
import csv
import io
import json
import math
import os
import socket
import time
import uuid

# -------------------- connection settings --------------------
# Loopback only: Unity connects to 127.0.0.1, and the protocol is unauthenticated.
HOST = '127.0.0.1'
PORT = 56001
SOCKET_TIMEOUT_SEC = float(os.environ.get("BO_SOCKET_TIMEOUT_SEC", "3600"))
SOCKET_ACCEPT_TIMEOUT_SEC = float(os.environ.get("BO_ACCEPT_TIMEOUT_SEC", "300"))
# Upper bound for one message line. The init message carries manual context embeddings
# (~16 KB per context at 1280 dimensions, ~51 KB at 4096), which the former 1 MiB limit
# rejected for larger context sets; Unity refuses to send an init larger than 64 MiB.
SOCKET_MAX_RECV_BUF_BYTES = int(os.environ.get("BO_MAX_RECV_BUF_BYTES", str(64 * 1024 * 1024)))
RECV_CHUNK_BYTES = 65536


class StopRequested(Exception):
    """Unity ended the study on purpose ({"type": "stop"}); the backend exits cleanly."""


def send_json_line(conn, obj):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        conn.sendall(line.encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        t = obj.get("type") if isinstance(obj, dict) else "unknown"
        raise ConnectionError(f"Failed to send message to Unity (type={t}): {e}") from e


def configure_listener_socket(s):
    """Socket options for the backend's single listening socket.

    On Windows, SO_REUSEADDR lets a second process bind a port that is still being listened
    on, so a stale backend could receive Unity's connection; SO_EXCLUSIVEADDRUSE refuses that
    and still allows an immediate restart. On POSIX, SO_REUSEADDR only permits rebinding
    while an earlier connection lingers in TIME_WAIT, which a quick restart needs.
    """
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def accept_unity_connection(s, accept_timeout_sec):
    """Listen on HOST:PORT and return Unity's connection; the listener is closed after it.

    PythonStarter.cs waits for the "Server starts, waiting for connection..." line before
    Unity connects, so that line must stay verbatim.
    """
    if accept_timeout_sec <= 0:
        raise ValueError(f"BO_ACCEPT_TIMEOUT_SEC must be > 0, got {accept_timeout_sec}")
    s.settimeout(accept_timeout_sec)
    try:
        s.bind((HOST, PORT))
    except OSError as e:
        raise OSError(
            f"Port {PORT} is already in use, most likely by an optimizer backend left over "
            "from an earlier session. End that python process (Task Manager / Activity "
            f"Monitor) and start again. ({e})"
        ) from e
    s.listen(1)
    print('Server starts, waiting for connection...', flush=True)
    try:
        conn, addr = s.accept()
    except socket.timeout as e:
        raise TimeoutError(f"Socket accept timed out after {accept_timeout_sec} seconds.") from e
    s.close()  # one Unity client per run: accept no further connections
    print('Connected by', addr, flush=True)
    return conn


def receive_init_message(conn, socket_timeout_sec):
    """Arm the session timeout, start a fresh reader and block for Unity's init message.

    Messages before the init (non-dict lines, other types) are skipped. Unity sends a stop
    instead of the init when it refuses to start the study (invalid configuration).
    """
    if socket_timeout_sec <= 0:
        raise ValueError(f"BO_SOCKET_TIMEOUT_SEC must be > 0, got {socket_timeout_sec}")
    conn.settimeout(socket_timeout_sec)
    reset_receive_state()
    while True:
        msg = recv_json_message(conn)
        if msg is None:
            raise RuntimeError("Did not receive init message.")
        if isinstance(msg, dict) and msg.get("type") == "init":
            return msg
        _raise_if_stop(msg)


def _raise_if_stop(msg):
    if isinstance(msg, dict) and msg.get("type") == "stop":
        reason = str(msg.get("reason") or "").strip() or "no reason given"
        print(f"Stop requested by Unity: {reason}", flush=True)
        raise StopRequested(reason)


def close_connection(conn):
    if conn is None:
        return
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

# -------------------- NDJSON receiving --------------------
class _ReceiveBuffer:
    """Unread bytes of the Unity connection, kept across recv_json_message() calls.

    Lines are split on the raw byte b"\\n" before decoding: 0x0A never occurs inside a
    multi-byte UTF-8 sequence, so a character split between two recv() calls ("Größe") is
    reassembled before it is decoded, and the size limit counts real bytes. Only the
    unterminated tail is joined when its line completes, so a large init message is
    received in linear time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.lines = collections.deque()  # complete lines (bytes, without the newline)
        self.partial = []                 # chunks of the unterminated last line
        self.partial_len = 0

    def feed(self, chunk):
        cut = chunk.rfind(b"\n")
        if cut < 0:
            self.partial.append(chunk)
            self.partial_len += len(chunk)
        else:
            head = b"".join(self.partial) + chunk[:cut]
            self.lines.extend(head.split(b"\n"))
            tail = chunk[cut + 1:]
            self.partial = [tail] if tail else []
            self.partial_len = len(tail)
        if self.partial_len > SOCKET_MAX_RECV_BUF_BYTES:
            preview = _decode(b"".join(self.partial)[-200:]).replace("\n", "\\n")
            self.reset()
            raise RuntimeError(
                f"Socket receive buffer exceeded {SOCKET_MAX_RECV_BUF_BYTES} bytes without a newline; "
                "possible framing error or oversized message (the limit is the environment variable "
                f"BO_MAX_RECV_BUF_BYTES). Tail preview: {preview}"
            )

    def pending_bytes(self):
        return b"".join(line + b"\n" for line in self.lines) + b"".join(self.partial)


def _decode(data):
    return data.decode("utf-8", errors="replace")


_RECEIVE_BUFFER = _ReceiveBuffer()


def reset_receive_state():
    """Forget unread bytes (a new connection starts with an empty reader)."""
    _RECEIVE_BUFFER.reset()


def pending_receive_buffer():
    """Received but not yet consumed text, complete lines and the unterminated tail."""
    return _decode(_RECEIVE_BUFFER.pending_bytes())


def recv_json_message(conn, awaiting_objectives=False):
    """Receive one NDJSON message while preserving unread bytes across calls.

    Returns None once Unity closes the connection. A malformed line is skipped with a
    warning, except while ``awaiting_objectives``: then it raises at once (see
    recv_objectives_blocking).
    """
    while True:
        if _RECEIVE_BUFFER.lines:
            line = _decode(_RECEIVE_BUFFER.lines.popleft()).rstrip("\r")
            if not line.strip():
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                preview = line[:200]
                if awaiting_objectives:
                    raise RuntimeError(
                        f"Received a malformed JSON line from Unity while waiting for objectives: {e}. "
                        f"Payload preview: {preview!r}. If it was the objectives reply, Unity is already "
                        f"waiting for the next design, so the backend stops here instead of waiting "
                        f"{SOCKET_TIMEOUT_SEC} seconds for objectives that will not arrive."
                    ) from e
                # Keep the reader tolerant to non-critical malformed lines.
                print(
                    f"Warning: skipping malformed JSON line from Unity: {e}. Payload preview: {preview!r}",
                    flush=True,
                )
                continue
        try:
            chunk = conn.recv(RECV_CHUNK_BYTES)
        except socket.timeout as e:
            raise TimeoutError(f"Socket receive timed out after {SOCKET_TIMEOUT_SEC} seconds.") from e
        if not chunk:
            # NDJSON requires newline framing. On close, remaining bytes are trailing partial data.
            trailing = _decode(b"".join(_RECEIVE_BUFFER.partial)).strip()
            _RECEIVE_BUFFER.reset()
            if trailing:
                print("Warning: discarding trailing unterminated socket data:",
                      trailing[-200:].replace("\n", "\\n"), flush=True)
            return None
        _RECEIVE_BUFFER.feed(chunk)


def ndjson_reader(conn):
    while True:
        msg = recv_json_message(conn)
        if msg is None:
            return
        yield msg


def recv_objectives_blocking(conn):
    """Block for Unity's next 'objectives' message and return its 'values' dict.

    Returns None when Unity closes the connection. Other message types are skipped (future
    Unity status messages must not break a backend), except 'stop', which raises
    StopRequested. A malformed line raises at once: if it was the objectives reply, Unity
    already waits for the next design and both sides would block until the socket timeout.
    """
    while True:
        msg = recv_json_message(conn, awaiting_objectives=True)
        if msg is None:
            return None
        if not isinstance(msg, dict):
            continue
        _raise_if_stop(msg)
        if msg.get("type") != "objectives":
            continue
        values = msg.get("values")
        if not isinstance(values, dict):
            raise RuntimeError("Received malformed 'objectives' message: missing or non-dict 'values'.")
        return values

# -------------------- init message parsing --------------------
def parse_param_init(init_val):
    # Accept typed JSON: {"low": ..., "high": ...}
    if isinstance(init_val, dict):
        if "low" not in init_val or "high" not in init_val:
            raise ValueError(f"Parameter init parse error (missing 'low'/'high'): {init_val}")
        return float(init_val["low"]), float(init_val["high"])
    parts = [p.strip() for p in str(init_val).split(",")]
    if len(parts) < 2:
        raise ValueError(f"Parameter init parse error: '{init_val}'")
    return float(parts[0]), float(parts[1])


def parse_obj_init(init_val):
    # Accept typed JSON: {"low": ..., "high": ..., "minimize": 0/1}
    if isinstance(init_val, dict):
        if "low" not in init_val or "high" not in init_val:
            raise ValueError(f"Objective init parse error (missing 'low'/'high'): {init_val}")
        if "minimize" not in init_val:
            raise ValueError(f"Objective init parse error (missing 'minimize'): {init_val}")
        return float(init_val["low"]), float(init_val["high"]), int(init_val["minimize"])
    parts = [p.strip() for p in str(init_val).split(",")]
    if len(parts) < 3:
        raise ValueError(f"Objective init parse error: '{init_val}'")
    return float(parts[0]), float(parts[1]), int(float(parts[2]))


def get_cfg_int(cfg, key, default=None, required=False):
    if key in cfg and cfg.get(key) is not None:
        try:
            return int(cfg.get(key))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Config field '{key}' must be an integer, got {cfg.get(key)!r}") from e
    if required:
        raise ValueError(f"Missing required config field '{key}'")
    return int(default) if default is not None else None


def get_cfg_float(cfg, key, default=None, required=False):
    if key in cfg and cfg.get(key) is not None:
        try:
            return float(cfg.get(key))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Config field '{key}' must be a number, got {cfg.get(key)!r}") from e
    if required:
        raise ValueError(f"Missing required config field '{key}'")
    return float(default) if default is not None else None


def get_cfg_bool(cfg, key, default=None, required=False):
    if key in cfg and cfg.get(key) is not None:
        val = cfg.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)) and float(val) in (0.0, 1.0):
            return bool(val)
        if isinstance(val, str):
            token = val.strip().lower()
            if token in ("true", "1"):
                return True
            if token in ("false", "0"):
                return False
        raise ValueError(f"Config field '{key}' must be a boolean, got {cfg.get(key)!r}")
    if required:
        raise ValueError(f"Missing required config field '{key}'")
    return default


def get_cfg_str(cfg, key, default=""):
    val = cfg.get(key)
    if val is None:
        return default
    token = str(val).strip().lower()
    return token if token else default


def normalize_user_token(value, default="-1"):
    token = str(value).strip() if value is not None else ""
    return token if token else default


def normalize_log_folder_token(value, default="-1"):
    token = normalize_user_token(value, default=default)
    invalid_chars = set('/\\:*?"<>|')
    cleaned_chars = []
    for ch in token:
        if ch in invalid_chars or ord(ch) < 32:
            cleaned_chars.append("_")
        else:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars).strip().strip(".")
    if cleaned in ("", ".", ".."):
        return default
    return cleaned


def parse_user_ids(init_msg):
    """(USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID) from the init message.

    The IDs are logged verbatim; the *_LOG_ID variants name the log folders and have path
    characters replaced, which is announced because the folder then differs from the ID.
    """
    user = init_msg.get("user", {}) or {}
    user_id = normalize_user_token(user.get("userId"), default="-1")
    condition_id = normalize_user_token(user.get("conditionId"), default="-1")
    group_id = normalize_user_token(user.get("groupId"), default="-1")
    user_log_id = normalize_log_folder_token(user_id, default="-1")
    condition_log_id = normalize_log_folder_token(condition_id, default="-1")
    if user_log_id != user_id:
        print(
            f"Warning: userId '{user_id}' was normalized to safe log-folder token '{user_log_id}'.",
            flush=True,
        )
    if condition_log_id != condition_id:
        print(
            f"Warning: conditionId '{condition_id}' was normalized to safe log-folder token '{condition_log_id}'.",
            flush=True,
        )
    return user_id, condition_id, group_id, user_log_id, condition_log_id

# -------------------- init validation --------------------
# Checked at init, before the sampling phase: the same settings otherwise fail only at the
# first model fit or acquisition, after the participant has completed the sampling phase.
def validate_parameter_bounds(names, bounds, backend):
    """Every parameter needs finite bounds with low < high (``bounds`` items start lo, hi)."""
    for name, info in zip(names, bounds):
        lo, hi = info[0], info[1]
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"Parameter '{name}' bounds must be finite, got ({lo}, {hi})")
        if hi < lo:
            raise ValueError(f"Parameter '{name}' has invalid bounds: low={lo} > high={hi}")
        if hi == lo:
            raise ValueError(
                f"Parameter '{name}' has a degenerate range [{lo}, {hi}]. "
                f"The {backend} backend needs a non-empty range on every parameter."
            )


def validate_objective_bounds(names, objectives):
    """Finite objective bounds with low <= high and a 0/1 minimize flag (items start lo, hi, flag)."""
    for name, info in zip(names, objectives):
        lo, hi, minflag = info[0], info[1], info[2]
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"Objective '{name}' bounds must be finite, got ({lo}, {hi})")
        if hi < lo:
            raise ValueError(f"Objective '{name}' has invalid bounds: low={lo} > high={hi}")
        if int(minflag) not in (0, 1):
            raise ValueError(f"Objective '{name}' minimize flag must be 0 or 1, got {minflag}")


def validate_raw_samples(num_restarts, raw_samples):
    """BoTorch's optimize_acqf picks its restart points among the raw samples."""
    if raw_samples < num_restarts:
        raise ValueError(
            f"rawSamples ({raw_samples}) must be >= numRestarts ({num_restarts}): the acquisition "
            f"optimizer starts its {num_restarts} restarts from the best of the {raw_samples} raw "
            "samples, and would only fail after the sampling phase. Raise rawSamples or lower numRestarts."
        )

# -------------------- run folders --------------------
def get_unique_folder(parent, folder_name):
    base_path = os.path.join(parent, folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path
    if os.path.isdir(base_path):
        visible_entries = [
            name for name in os.listdir(base_path)
            if name != ".DS_Store" and not name.endswith(".meta")
        ]
        if not visible_entries:
            return base_path
    k = 1
    while True:
        p = os.path.join(parent, f"{folder_name}_{k}")
        if not os.path.exists(p):
            os.makedirs(p)
            return p
        k += 1

# -------------------- run logs: atomic, lock-tolerant writes --------------------
# The backends rewrite ObservationsPerEvaluation.csv after every evaluation and append to the
# other logs. On Windows a log that is open in Excel cannot be written or replaced
# (PermissionError), which used to end the participant's session, and a crash during an
# in-place rewrite truncated the study log. Whole-file writes therefore go through a temp file
# in the same folder + fsync + os.replace (readers see the old or the new file, never half of
# one; in place only where the replace alone is refused, see _rewrite_file), and every write is
# retried for LOG_WRITE_RETRY_SEC. A log that stays locked is
# reported once and kept in memory: reads in this process (read_log_text, read_observation_log)
# return the unsaved content, so read-modify-write rewrites lose no row, and the next
# successful write brings the file up to date while the study goes on. Until then its complete
# content is also kept next to it as <name>.unsaved<ext> (Unity kills the backend when a
# session is aborted, which would lose what is only in memory). flush_pending_logs() retries
# at shutdown and otherwise leaves that copy.
# All logs are UTF-8 explicitly: the locale default on Windows is cp1252.
LOG_WRITE_RETRY_SEC = 2.0
_UNSAVED_TEXT = {}     # abs path -> complete content the file should have
_UNSAVED_APPENDS = {}  # abs path -> [text] still to be appended to the file on disk


def _log_key(path):
    return os.path.abspath(path)


def _with_retries(action, retry_sec):
    """Run ``action`` until it succeeds or ``retry_sec`` is used up; return the last OSError."""
    deadline = time.monotonic() + retry_sec
    delay = 0.05
    while True:
        try:
            action()
            return None
        except OSError as e:
            if time.monotonic() + delay > deadline:
                return e
            time.sleep(delay)
            delay = min(2 * delay, 0.5)


def _replace_file(path, text):
    folder = os.path.dirname(path) or "."
    # Dot-prefixed: Unity's asset importer ignores the temp file (LogData is under Assets/).
    # Created like any log (0o666 & ~umask); tempfile.mkstemp would leave the log at 0o600.
    tmp_path = os.path.join(folder, f".{os.path.basename(path)}.{uuid.uuid4().hex[:12]}.tmp")
    f = open(tmp_path, "x", encoding="utf-8", newline="")
    try:
        with f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _overwrite_file(path, text):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(text)


def _rewrite_file(path, text):
    """Replace ``path`` atomically; in place when the system refuses only the replace.

    Windows refuses to replace a file that another program holds open without delete sharing
    (a log viewer, a script reading the log) although that program lets it be written, as the
    former in-place rewrite did. A program that blocks writing too (Excel) fails both ways,
    and the log is kept in memory (_write_log_text).
    """
    try:
        _replace_file(path, text)
    except PermissionError:
        _overwrite_file(path, text)


def _append_file(path, text):
    with open(path, "a", newline="", encoding="utf-8") as f:
        f.write(text)


def _unsaved_copy_path(path):
    root, ext = os.path.splitext(path)
    return f"{root}.unsaved{ext}"


def _save_unsaved_copy(key, text):
    """Keep a locked log's complete content next to it; True if that worked (one attempt)."""
    if text is None:
        return False
    try:
        _replace_file(_unsaved_copy_path(key), text)
        return True
    except OSError:
        return False


def _drop_unsaved_copy(key):
    try:
        os.remove(_unsaved_copy_path(key))
    except OSError:
        pass


def _disk_text(path):
    """A log's content on disk ("" if it does not exist), or None if it cannot be read now."""
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except OSError:
        return None


def _report_write(path, error, was_unsaved, copied=False):
    if error is not None and not was_unsaved:
        copy = f" and as {_unsaved_copy_path(path)}" if copied else ""
        print(
            f"Warning: could not write {path} ({error}). Is it open in another program (e.g. Excel)? "
            f"The study continues: the log is kept in memory{copy} and saved at the next successful "
            "write. Close the file to let the backend save it.",
            flush=True,
        )
    elif error is None and was_unsaved:
        print(f"Saved {path} again; the file is complete.", flush=True)


def _write_log_text(path, text, retry_sec):
    key = _log_key(path)
    was_unsaved = key in _UNSAVED_TEXT or key in _UNSAVED_APPENDS
    error = _with_retries(lambda: _rewrite_file(path, text), retry_sec)
    _UNSAVED_APPENDS.pop(key, None)  # the complete content supersedes pending appends
    copied = False
    if error is None:
        _UNSAVED_TEXT.pop(key, None)
        if was_unsaved:
            _drop_unsaved_copy(key)
    else:
        _UNSAVED_TEXT[key] = text
        copied = _save_unsaved_copy(key, text)
    _report_write(path, error, was_unsaved, copied)
    return error is None


def _append_log_text(path, text, retry_sec):
    key = _log_key(path)
    if key in _UNSAVED_TEXT:
        return _write_log_text(path, _UNSAVED_TEXT[key] + text, retry_sec)
    was_unsaved = key in _UNSAVED_APPENDS
    pending = "".join(_UNSAVED_APPENDS.get(key, [])) + text
    if not pending:
        return True
    error = _with_retries(lambda: _append_file(path, pending), retry_sec)
    copied = False
    if error is None:
        _UNSAVED_APPENDS.pop(key, None)
        if was_unsaved:
            _drop_unsaved_copy(key)
    else:
        _UNSAVED_APPENDS[key] = [pending]
        on_disk = _disk_text(path)
        copied = _save_unsaved_copy(key, None if on_disk is None else on_disk + pending)
    _report_write(path, error, was_unsaved, copied)
    return error is None


def _retry_budget(path):
    # A log already known to be locked gets one attempt per write, so a file left open in
    # Excel delays the study once (LOG_WRITE_RETRY_SEC), not at every evaluation.
    key = _log_key(path)
    return 0.0 if (key in _UNSAVED_TEXT or key in _UNSAVED_APPENDS) else LOG_WRITE_RETRY_SEC


def write_log_text(path, text):
    """Replace ``path``'s content with ``text`` (atomic; retried; kept in memory if locked)."""
    return _write_log_text(path, text, _retry_budget(path))


def append_log_text(path, text):
    """Append ``text`` to ``path`` (retried; kept in memory if locked)."""
    return _append_log_text(path, text, _retry_budget(path))


def read_log_text(path):
    """A log's current content, unsaved writes included; None if it does not exist."""
    key = _log_key(path)
    if key in _UNSAVED_TEXT:
        return _UNSAVED_TEXT[key]
    appended = "".join(_UNSAVED_APPENDS.get(key, []))
    if not os.path.exists(path):
        return appended or None
    content = []

    def _read():
        with open(path, "r", newline="", encoding="utf-8") as f:
            content[:] = [f.read()]

    error = _with_retries(_read, LOG_WRITE_RETRY_SEC)
    if error is not None:
        raise error
    return content[0] + appended


def log_exists(path):
    key = _log_key(path)
    return key in _UNSAVED_TEXT or key in _UNSAVED_APPENDS or os.path.exists(path)


def _log_is_empty(path):
    key = _log_key(path)
    if key in _UNSAVED_TEXT:
        return _UNSAVED_TEXT[key] == ""
    if key in _UNSAVED_APPENDS:
        return False
    return not os.path.exists(path) or os.path.getsize(path) == 0


def unsaved_logs():
    """Paths whose on-disk content is behind (for diagnostics and tests)."""
    return sorted(set(_UNSAVED_TEXT) | set(_UNSAVED_APPENDS))


def discard_unsaved_logs():
    """Forget unsaved log content without writing it (tests)."""
    _UNSAVED_TEXT.clear()
    _UNSAVED_APPENDS.clear()


def flush_pending_logs():
    """Try every unsaved log once more; save what stays locked as '<name>.unsaved<ext>'.

    Called when a backend shuts down, so no evaluation is lost because a log was open in
    another program at the end of the session. No retry wait here: these logs were already
    locked at their last write, and Unity terminates a backend that does not exit promptly
    after a stop request. Returns the paths that are still behind.
    """
    for key in unsaved_logs():
        if key in _UNSAVED_TEXT:
            ok = _write_log_text(key, _UNSAVED_TEXT[key], 0.0)
        else:
            ok = _append_log_text(key, "", 0.0)
        if ok:
            continue
        fallback = _unsaved_copy_path(key)
        if key in _UNSAVED_TEXT:
            content, what = _UNSAVED_TEXT[key], "its complete content"
        else:
            # No read retries either (read_log_text waits up to LOG_WRITE_RETRY_SEC).
            on_disk = _disk_text(key)
            pending = "".join(_UNSAVED_APPENDS.get(key, []))
            if on_disk is None:
                content, what = pending, "the rows missing from it"
            else:
                content, what = on_disk + pending, "its complete content"
        try:
            _replace_file(fallback, content or "")
        except OSError as e:
            print(f"ERROR: {key} is still locked and {fallback} could not be written ({e}); "
                  "the run's last rows are not on disk.", flush=True)
            continue
        _UNSAVED_TEXT.pop(key, None)
        _UNSAVED_APPENDS.pop(key, None)
        print(f"Warning: {key} is still locked; {what} was saved as {fallback} instead.", flush=True)
    return unsaved_logs()

# -------------------- CSV logs (';'-separated, UTF-8) --------------------
def _csv_text(rows, header=None):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, delimiter=';')
    if header is not None:
        w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def write_csv_rows(path, header, rows):
    """Rewrite a CSV log with ``header`` and ``rows`` (lists)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return write_log_text(path, _csv_text(rows, header))


def append_csv_rows(path, rows, header=None):
    """Append ``rows`` (lists) to a CSV log, preceded by ``header`` while the log is empty."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if header is not None and not _log_is_empty(path):
        header = None
    return append_log_text(path, _csv_text(rows, header))


def write_dataframe_csv(path, df):
    """Rewrite a CSV log from a DataFrame, byte for byte as df.to_csv(path, sep=';', index=False)."""
    return write_log_text(path, df.to_csv(sep=';', index=False))


def read_observation_log(path):
    """Read ObservationsPerEvaluation.csv back without re-typing any cell.

    The log is rewritten after every evaluation; type inference would turn IDs such as
    "007" or "NA" into 7 or an empty cell in every earlier row, while FinalDesignSelector
    and the questionnaire logs match IDs as exact strings.
    """
    import pandas as pd  # lazy: the protocol helpers need the standard library only

    text = read_log_text(path)
    if text is None:
        raise FileNotFoundError(f"Log file not found: {path}")
    return pd.read_csv(io.StringIO(text), delimiter=';', dtype=str, keep_default_na=False)


def create_csv_file(csv_file_path, fieldnames):
    append_csv_rows(csv_file_path, [], header=fieldnames)


def write_data_to_csv(csv_file_path, fieldnames, rows):
    buf = io.StringIO(newline="")
    csv.DictWriter(buf, fieldnames=fieldnames, delimiter=';').writerows(rows)
    append_log_text(csv_file_path, buf.getvalue())
