#!/usr/bin/env python3
"""Assemble the Unity Asset Store upload project for Bayesian Optimization for Unity (BOforUnity).

The repository stays the single source of truth. This script copies the package content into a clean Unity project
(default: AssetStore/UploadProject), arranged the way the Asset Store expects it, and checks it against the
submission rules that can be checked offline. Open that project in Unity and upload from there (see README.md).

What it does:
  * copies the git-tracked (and new, not ignored) files of Assets/BOforUnity, Assets/QuestionnaireToolkit and
    Assets/StreamingAssets/BOData, with their .meta files (GUIDs are kept, so scenes and prefabs stay wired);
  * moves QuestionnaireToolkit below the single package root Assets/BOforUnity (rule 2.1.a);
  * leaves out the Python installers and installer scripts (rule 1.5.a: no executables) and, unless --with-tests,
    the test assemblies;
  * adds AssetStore/PackageExtras (Third-Party Notices, ...) and renders the documentation to self-contained,
    offline HTML (README.md, docs/*.md, AssetStore/Docs/*.md) into Assets/BOforUnity/Documentation;
  * gives every added file and folder a .meta with a GUID derived from its path, so store updates replace files
    instead of duplicating them;
  * writes a minimal Packages/manifest.json for the chosen Unity editor (its built-in package versions), so the
    project behaves like a buyer's fresh project instead of inheriting this repository's settings;
  * validates the result (single root folder, .meta files, path lengths, forbidden file types, namespaces,
    hard-coded paths, open TODOs in the notices) and, with --verify, compiles it in Unity batch mode
    (with --with-tests --verify it also runs the EditMode tests).

Requires Python 3.9+ and, for the documentation, markdown-it-py and mdit-py-plugins:
    python3 -m pip install markdown-it-py mdit-py-plugins
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import html
import json
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "AssetStore"
EXTRAS = HERE / "PackageExtras"
DOCS = HERE / "Docs"
DEFAULT_OUT_NAME = "UploadProject"
# Used instead of AssetStore/ when the repository lies in iCloud Drive (macOS "Desktop & Documents" sync): iCloud
# evicts freshly written files under storage pressure, and a Unity project's Library/ must not sync anyway.
ICLOUD_FALLBACK_DIR = Path.home() / "BOforUnity-AssetStore"
MARKER = ".boforunity-upload-project"

PACKAGE_ROOT = "Assets/BOforUnity"
DOC_DIR = PACKAGE_ROOT + "/Documentation"
GITHUB = "https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity"
GITHUB_BLOB = GITHUB + "/blob/main/"
GITHUB_RAW = "https://raw.githubusercontent.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/main/"

# First matching prefix wins. None = not part of the store package.
TESTS_PREFIXES = ("Assets/BOforUnity/Tests/", "Assets/BOforUnity/Tests.meta")
MAPPING = [
    ("Assets/BOforUnity/", "Assets/BOforUnity/"),
    ("Assets/BOforUnity.meta", "Assets/BOforUnity.meta"),
    ("Assets/QuestionnaireToolkit/", "Assets/BOforUnity/QuestionnaireToolkit/"),
    ("Assets/QuestionnaireToolkit.meta", "Assets/BOforUnity/QuestionnaireToolkit.meta"),
    # Python installers (.exe/.pkg) and the scripts that run them: the store accepts no executables (1.5.a).
    # PythonStarter then tells the user to install Python 3.13 instead (README of this folder).
    ("Assets/StreamingAssets/BOData/Installation/Windows", None),
    ("Assets/StreamingAssets/BOData/Installation/MacOs", None),
    ("Assets/StreamingAssets/BOData/Installation/Linux", None),
    ("Assets/StreamingAssets/BOData/LogData", None),  # runtime output; only its folder .meta is tracked
    ("Assets/StreamingAssets/BOData/", "Assets/StreamingAssets/BOData/"),
    ("Assets/StreamingAssets/BOData.meta", "Assets/StreamingAssets/BOData.meta"),
    ("Assets/StreamingAssets.meta", "Assets/StreamingAssets.meta"),
]

# Rendered into Assets/BOforUnity/Documentation (source relative to the repository -> output file name).
DOCUMENTS = [
    ("AssetStore/Docs/GettingStarted.md", "GettingStarted.html", "BOforUnity - Getting Started"),
    ("README.md", "Manual.html", "BOforUnity - Manual"),
    ("docs/dbo-backend.md", "DBO-Backend.html", "BOforUnity - Dynamic BO (DBO) Backend"),
    ("docs/meta-taf-student-guide.md", "MetaTAF-Guide.html", "BOforUnity - MetaTAF Guide"),
]
MANUAL_PREFACE = "AssetStore/Docs/ManualPreface.md"

# Modules the package and its example scenes need, plus what Unity adds to every new project anyway.
REQUIRED_PACKAGES = {"com.unity.nuget.newtonsoft-json": "3.2.2"}
BUILT_IN_FROM_EDITOR = ("com.unity.ugui", "com.unity.test-framework")

FORBIDDEN_EXTENSIONS = {
    ".exe", ".msi", ".pkg", ".dmg", ".apk", ".aab", ".app", ".bat", ".cmd", ".sh", ".ps1", ".com", ".jar",
    ".zip", ".7z", ".rar", ".unitypackage", ".mp4", ".mov", ".avi", ".webm", ".js",
}
SKIP_NAMES = {".DS_Store", "Thumbs.db", ".gitkeep"}
MAX_PATH = 150  # 2.1.e: measured from "Assets/" for .unitypackage submissions
TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".csv", ".xml", ".bytes", ".yaml", ".yml", ".py"}


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────

def fail(message: str) -> None:
    print("error: " + message, file=sys.stderr)
    sys.exit(1)


def stable_guid(dest: str) -> str:
    return hashlib.md5(("BOforUnity/AssetStore:" + dest).encode("utf-8")).hexdigest()


def meta_for(dest: str, is_dir: bool) -> str:
    guid = stable_guid(dest)
    if is_dir:
        return (f"fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n"
                "  userData: \n  assetBundleName: \n  assetBundleVariant: \n")
    importer = "TextScriptImporter" if Path(dest).suffix.lower() in TEXT_EXTENSIONS else "DefaultImporter"
    return (f"fileFormatVersion: 2\nguid: {guid}\n{importer}:\n  externalObjects: {{}}\n"
            "  userData: \n  assetBundleName: \n  assetBundleVariant: \n")


def map_path(src: str, with_tests: bool) -> str | None:
    if not with_tests and src.startswith(TESTS_PREFIXES):
        return None
    for prefix, dest in MAPPING:
        if src == prefix or src.startswith(prefix):
            return None if dest is None else dest + src[len(prefix):]
    return None


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "Assets"],
            capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as ex:
        fail(f"could not list the repository files with git: {ex}")
    return sorted({p for p in out.decode("utf-8").split("\0") if p})


def in_icloud_drive(path: Path) -> bool:
    if sys.platform != "darwin":
        return False
    cloud = Path.home() / "Library" / "Mobile Documents"
    roots = [cloud.resolve()]
    for name in ("Desktop", "Documents"):  # symlinks to ~/Desktop and ~/Documents when their sync is on
        link = cloud / "com~apple~CloudDocs" / name
        if link.is_symlink():
            roots.append(link.resolve())
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def default_out(with_tests: bool) -> Path:
    name = DEFAULT_OUT_NAME + ("-tests" if with_tests else "")
    if in_icloud_drive(HERE):
        print(f"note: the repository is in iCloud Drive; writing the Unity project to {ICLOUD_FALLBACK_DIR} instead "
              "(iCloud would sync its Library/ folder and evict freshly written files). Use --out to choose another folder.")
        return ICLOUD_FALLBACK_DIR / name
    return HERE / name


def version_key(version: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"[.a-z]+", version) if p != "")


# ── Unity editor discovery ─────────────────────────────────────────────────────────────────────

def editor_layouts(root: Path) -> tuple[Path, Path]:
    """(executable, BuiltInPackages dir) of an editor installed at root (a Hub '<version>' folder)."""
    mac = root / "Unity.app" / "Contents"
    if mac.is_dir():
        return mac / "MacOS" / "Unity", mac / "Resources" / "PackageManager" / "BuiltInPackages"
    data = root / "Editor" / "Data" / "Resources" / "PackageManager" / "BuiltInPackages"
    exe = root / "Editor" / ("Unity.exe" if os.name == "nt" else "Unity")
    return exe, data


def installed_editors() -> dict[str, Path]:
    hubs = [Path("/Applications/Unity/Hub/Editor"), Path.home() / "Applications" / "Unity" / "Hub" / "Editor",
            Path("C:/Program Files/Unity/Hub/Editor"), Path.home() / "Unity" / "Hub" / "Editor"]
    found = {}
    for hub in hubs:
        if hub.is_dir():
            for child in hub.iterdir():
                exe, builtin = editor_layouts(child)
                if builtin.is_dir():
                    found.setdefault(child.name, child)
    return found


def choose_editor(requested: str | None, editor_path: str | None) -> tuple[str, Path]:
    if editor_path:
        root = Path(editor_path).expanduser().resolve()
        if root.name.endswith(".app"):
            root = root.parent
        if not editor_layouts(root)[1].is_dir():
            fail(f"{editor_path} does not look like a Unity editor installation (no BuiltInPackages folder).")
        return requested or root.name, root
    editors = installed_editors()
    if requested:
        if requested not in editors:
            fail(f"Unity {requested} is not installed in a Unity Hub folder (found: {', '.join(sorted(editors)) or 'none'}). "
                 "Install it or pass --editor-path.")
        return requested, editors[requested]
    # 1.3.c: submissions made with Unity 6.5 or newer must support URP/HDRP; the example scenes use the Built-in RP.
    eligible = sorted((v for v in editors if version_key(v) < version_key("6000.5")), key=version_key)
    if not eligible:
        fail("no Unity editor older than 6000.5 found in the Unity Hub folders "
             f"(found: {', '.join(sorted(editors)) or 'none'}). Install Unity 6000.4 (or 6000.3 LTS), "
             "or pass --editor-version/--editor-path.")
    return eligible[-1], editors[eligible[-1]]


def builtin_versions(editor_root: Path) -> dict[str, str]:
    builtin = editor_layouts(editor_root)[1]
    versions = {}
    for package_json in builtin.glob("*/package.json"):
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            versions[data["name"]] = data["version"]
        except (OSError, ValueError, KeyError):
            continue
    return versions


def write_manifest(out: Path, editor_root: Path) -> None:
    available = builtin_versions(editor_root)
    source = json.loads((REPO / "Packages" / "manifest.json").read_text(encoding="utf-8"))["dependencies"]
    deps = dict(REQUIRED_PACKAGES)
    for name in BUILT_IN_FROM_EDITOR:
        if name not in available:
            fail(f"the editor at {editor_root} has no built-in {name}.")
        deps[name] = available[name]
    for name in source:
        if name.startswith("com.unity.modules.") and name in available:
            deps[name] = available[name]
    manifest = {"dependencies": dict(sorted(deps.items()))}
    (out / "Packages").mkdir(parents=True, exist_ok=True)
    (out / "Packages" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ── documentation ──────────────────────────────────────────────────────────────────────────────

CSS = """
:root{color-scheme:light dark;--fg:#1f2328;--muted:#59636e;--bg:#ffffff;--code:#f6f8fa;--border:#d1d9e0;--link:#0969da;--note:#ddf4ff}
@media (prefers-color-scheme:dark){:root{--fg:#e6edf3;--muted:#9198a1;--bg:#0d1117;--code:#151b23;--border:#3d444d;--link:#4493f8;--note:#0c2d4b}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
a{color:var(--link)} h1,h2{border-bottom:1px solid var(--border);padding-bottom:.3em} h1,h2,h3,h4{line-height:1.25;margin-top:1.6em}
code,pre,.math{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em}
code{background:var(--code);padding:.15em .35em;border-radius:6px} pre{background:var(--code);padding:16px;border-radius:6px;overflow:auto}
pre code{background:none;padding:0} img{max-width:100%;height:auto} blockquote{margin:0;padding:0 1em;color:var(--muted);border-left:.25em solid var(--border)}
table{border-collapse:collapse;display:block;overflow:auto;margin:1em 0} th,td{border:1px solid var(--border);padding:6px 12px;vertical-align:top}
.math.block{display:block;text-align:center;margin:1em 0} .preface{background:var(--note);border:1px solid var(--border);border-radius:6px;padding:4px 16px;margin-bottom:24px}
nav.docs{font-size:.9em;color:var(--muted);margin-bottom:16px}
"""


def github_slug(text: str) -> str:
    slug = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return slug.replace(" ", "-")


def markdown_renderer():
    try:
        from markdown_it import MarkdownIt
        from mdit_py_plugins.anchors import anchors_plugin
        from mdit_py_plugins.dollarmath import dollarmath_plugin
    except ImportError:
        fail("rendering the documentation needs markdown-it-py and mdit-py-plugins:\n"
             "    python3 -m pip install markdown-it-py mdit-py-plugins")
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    md.use(anchors_plugin, min_level=1, max_level=6, slug_func=github_slug)
    md.use(dollarmath_plugin, allow_space=False, allow_digits=False)
    return md


def rewrite_references(rendered: str, source: str, doc_targets: dict[str, str]) -> str:
    """Inline local images as data URIs; point links to other rendered docs, everything else local to GitHub."""
    source_dir = posixpath.dirname(source)

    def resolve(ref: str) -> str | None:
        if re.match(r"^[a-z][a-z0-9+.-]*:", ref, re.I) or ref.startswith(("#", "//")):
            return None
        path = ref.split("#", 1)[0].split("?", 1)[0]
        return posixpath.normpath(posixpath.join(source_dir, path)) if path else None

    def replace(match: re.Match) -> str:
        attr, quote, ref = match.group(1), match.group(2), html.unescape(match.group(3))
        target = resolve(ref)
        if target is None:
            return match.group(0)
        anchor = "#" + ref.split("#", 1)[1] if "#" in ref else ""
        if attr == "src":
            file = REPO / target
            if file.suffix.lower() == ".gif" or not file.is_file():
                new = GITHUB_RAW + target  # animated demos are too large to inline; they load when online
            else:
                mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                new = f"data:{mime};base64," + base64.b64encode(file.read_bytes()).decode("ascii")
        elif target in doc_targets:
            new = doc_targets[target] + anchor
        else:
            new = GITHUB_BLOB + target + anchor
        return f'{attr}={quote}{html.escape(new, quote=True)}{quote}'

    return re.sub(r'\b(src|href)=(["\'])(.*?)\2', replace, rendered)


def render_documents(out: Path) -> list[str]:
    md = markdown_renderer()
    doc_targets = {src: name for src, name, _ in DOCUMENTS}
    written = []
    nav = " · ".join(f'<a href="{name}">{html.escape(title.split(" - ", 1)[1])}</a>' for _, name, title in DOCUMENTS)
    for source, name, title in DOCUMENTS:
        text = (REPO / source).read_text(encoding="utf-8")
        # GitHub's $`...`$ math form -> $...$ (the dollarmath plugin knows only the latter).
        text = re.sub(r"\$`([^`\n]+)`\$", r"$\1$", text)
        body = rewrite_references(md.render(text), source, doc_targets)
        if source == "README.md":
            preface = md.render((REPO / MANUAL_PREFACE).read_text(encoding="utf-8"))
            body = f'<div class="preface">{rewrite_references(preface, MANUAL_PREFACE, doc_targets)}</div>\n{body}'
        page = (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{html.escape(title)}</title>\n'
                f'<style>{CSS}</style>\n</head>\n<body>\n<main>\n<nav class="docs">{nav}</nav>\n{body}\n</main>\n</body>\n</html>\n')
        dest = f"{DOC_DIR}/{name}"
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        (out / dest).write_text(page, encoding="utf-8")
        written.append(dest)
    return written


# ── build ───────────────────────────────────────────────────────────────────────────────────────

def prepare_output(out: Path, clean: bool) -> None:
    if out == REPO or out in REPO.parents or (out / ".git").exists():
        fail(f"refusing to use {out} as the output folder.")
    if out.exists() and any(out.iterdir()) and not (out / MARKER).exists():
        fail(f"{out} exists and was not created by this script; choose another --out or empty it yourself.")
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("Generated by AssetStore/build_upload_project.py; safe to delete.\n", encoding="utf-8")
    # Keep Library/ (import cache) between builds unless --clean; the content folders are rebuilt from scratch.
    for name in ("Assets", "Packages", "ProjectSettings"):
        if (out / name).exists():
            shutil.rmtree(out / name)


def copy_package(out: Path, with_tests: bool) -> int:
    jobs = []
    for src in tracked_files():
        dest = map_path(src, with_tests)
        source = REPO / src
        if dest is None or not source.is_file() or Path(src).name in SKIP_NAMES:
            continue
        if Path(src).name.endswith(".meta") and Path(src[:-5]).name in SKIP_NAMES:
            continue
        jobs.append((source, out / dest))
    if EXTRAS.is_dir():
        for file in sorted(p for p in EXTRAS.rglob("*") if p.is_file() and p.name not in SKIP_NAMES):
            jobs.append((file, out / file.relative_to(EXTRAS)))
    for _, target in jobs:
        target.parent.mkdir(parents=True, exist_ok=True)
    # Contents only (no extended attributes or iCloud flags). In parallel: files that iCloud has evicted from the
    # repository download one request at a time otherwise.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda job: shutil.copyfile(*job), jobs))
    return len(jobs)


def inside_bundle(path: Path, assets: Path) -> bool:
    """Unity imports a macOS .bundle folder as one plugin asset: the files inside get no .meta of their own."""
    return any(parent.suffix == ".bundle" for parent in path.relative_to(assets).parents)


def add_missing_metas(out: Path) -> int:
    added = 0
    assets = out / "Assets"
    for path in sorted(assets.rglob("*")):
        if path.name.endswith(".meta") or path.name in SKIP_NAMES or inside_bundle(path, assets):
            continue
        meta = path.with_name(path.name + ".meta")
        if not meta.exists():
            dest = path.relative_to(out).as_posix()
            meta.write_text(meta_for(dest, path.is_dir()), encoding="utf-8")
            added += 1
    return added


# ── validation ──────────────────────────────────────────────────────────────────────────────────

def validate(out: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    assets = out / "Assets"

    top = sorted(p.name for p in assets.iterdir() if not p.name.endswith(".meta"))
    if top != ["BOforUnity", "StreamingAssets"]:
        errors.append(f"2.1.a: Assets/ must hold only BOforUnity and the special folder StreamingAssets, found {top}")
    streaming = sorted(p.name for p in (assets / "StreamingAssets").iterdir() if not p.name.endswith(".meta"))
    if streaming != ["BOData"]:
        errors.append(f"2.1.a: StreamingAssets/ must hold only BOData, found {streaming}")

    guids: dict[str, str] = {}
    total = 0
    for path in sorted(assets.rglob("*")):
        rel = path.relative_to(out).as_posix()
        if path.name.endswith(".meta"):
            if not path.with_name(path.name[:-5]).exists():
                errors.append(f"orphan .meta without its asset: {rel}")
            match = re.search(r"^guid: ([0-9a-f]{32})", path.read_text(encoding="utf-8", errors="replace"), re.M)
            if not match:
                errors.append(f"no GUID in {rel}")
            elif match.group(1) in guids:
                errors.append(f"duplicate GUID {match.group(1)} in {rel} and {guids[match.group(1)]}")
            else:
                guids[match.group(1)] = rel
            continue
        if not path.with_name(path.name + ".meta").exists() and not inside_bundle(path, assets):
            errors.append(f"missing .meta: {rel}")
        if len(rel) >= MAX_PATH:
            errors.append(f"2.1.e: path has {len(rel)} characters (limit {MAX_PATH}): {rel}")
        lowered = rel.lower()
        if "assetstoretools" in lowered or "__pycache__" in lowered or path.name in SKIP_NAMES:
            errors.append(f"file that must not ship: {rel}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"1.5.a/2.2/2.3.c: forbidden file type {path.suffix}: {rel}")
        if path.is_file():
            total += path.stat().st_size
            if path.suffix == ".cs":
                source = path.read_text(encoding="utf-8-sig", errors="replace")
                namespaces = re.findall(r"^\s*namespace\s+([\w.]+)", source, re.M)
                if not namespaces:
                    errors.append(f"2.5.a: no namespace: {rel}")
                for ns in namespaces:
                    if ns.split(".")[0] in ("Unity", "UnityEngine", "UnityEditor", "Ludiq"):
                        errors.append(f"2.5.a/5.1.b: namespace {ns} implies another entity: {rel}")
                if "Assets/QuestionnaireToolkit" in re.sub(r"//.*", "", source).replace('DefaultRoot = "Assets/QuestionnaireToolkit"', ""):
                    errors.append(f"hard-coded Assets/QuestionnaireToolkit path (the folder moves below {PACKAGE_ROOT}): {rel}")
                if re.search(r"BindingFlags\.NonPublic", source) and re.search(r"typeof\((EditorGUI|EditorGUIUtility|SerializedProperty|EditorWindow)\)|\"UnityEditor\.", source):
                    errors.append(f"2.5.g: reflection into internal Unity Editor API: {rel}")

    notices = out / PACKAGE_ROOT / "Third-Party Notices.txt"
    if not notices.is_file():
        errors.append("1.2.a: Third-Party Notices.txt is missing at the package root")
    else:
        for line in notices.read_text(encoding="utf-8").splitlines():
            if "TODO" in line:
                errors.append("1.2.a: open item in Third-Party Notices.txt: " + line.strip())
    if not (out / DOC_DIR).is_dir():
        errors.append("2.3.a: no offline documentation")
    if total > 6 * 1024 ** 3:
        errors.append("1.1.e: package larger than 6 GB")
    warnings.append(f"package content: {total / 1024 ** 2:.1f} MB in {len(guids)} assets")
    return errors, warnings


# ── Unity verification ──────────────────────────────────────────────────────────────────────────

def run_unity(out: Path, editor_root: Path, with_tests: bool) -> bool:
    exe = editor_layouts(editor_root)[0]
    log = out.parent / f"{out.name}-unity.log"
    args = [str(exe), "-batchmode", "-nographics", "-projectPath", str(out), "-logFile", str(log)]
    results = out.parent / f"{out.name}-editmode-results.xml"
    if with_tests:
        if results.exists():
            results.unlink()
        args += ["-runTests", "-testPlatform", "EditMode", "-testResults", str(results)]
    else:
        args += ["-quit"]
    print(f"Running Unity ({exe.parent.parent.name if exe.parent.name == 'MacOS' else exe.parent.name}); log: {log}")
    code = subprocess.run(args).returncode
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    package_paths = ("Assets/BOforUnity", "Assets/StreamingAssets")
    diagnostics = sorted({m.group(0).strip() for m in re.finditer(r"^.*(error|warning) CS\d{4}:.*$", text, re.M)
                          if m.group(0).lstrip().startswith(package_paths)})
    for line in diagnostics:
        print("  " + line)
    compile_errors = [d for d in diagnostics if " error CS" in d]
    print(f"Unity exit code {code}; {len(compile_errors)} compiler error(s), "
          f"{len(diagnostics) - len(compile_errors)} warning(s) from package scripts.")
    ok = not compile_errors and ("Scripts have compiler errors" not in text)
    if with_tests:
        if results.exists():
            run = ET.parse(results).getroot()
            print(f"EditMode tests: {run.get('passed')} passed, {run.get('failed')} failed, "
                  f"{run.get('skipped')} skipped of {run.get('total')} ({results})")
            for case in run.iter("test-case"):
                if case.get("result") == "Failed":
                    message = case.findtext("failure/message") or ""
                    print(f"  FAILED {case.get('fullname')}: {message.strip()[:300]}")
            ok = ok and run.get("failed") == "0"
        else:
            print("EditMode test results were not written; see the log.")
            ok = False
    else:
        ok = ok and code == 0
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="output Unity project folder (default: AssetStore/UploadProject, or "
                        "UploadProject-tests with --with-tests; below ~/BOforUnity-AssetStore if the repository is in iCloud Drive)")
    parser.add_argument("--editor-version", help="Unity editor version to target, e.g. 6000.4.7f1 "
                        "(default: newest installed version below 6000.5)")
    parser.add_argument("--editor-path", help="Unity editor installation folder, if not in a Unity Hub location")
    parser.add_argument("--with-tests", action="store_true",
                        help="include Assets/BOforUnity/Tests (for local checks; never upload this variant)")
    parser.add_argument("--verify", action="store_true",
                        help="open the project in Unity batch mode and check that it compiles "
                             "(with --with-tests: run the EditMode tests)")
    parser.add_argument("--clean", action="store_true", help="also delete the Unity import cache (Library/)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when piped or logged

    out = Path(args.out).expanduser().resolve() if args.out else default_out(args.with_tests)
    version, editor_root = choose_editor(args.editor_version, args.editor_path)
    print(f"Target editor: Unity {version} ({editor_root})")

    prepare_output(out, args.clean)
    copied = copy_package(out, args.with_tests)
    docs = render_documents(out)
    metas = add_missing_metas(out)
    write_manifest(out, editor_root)
    (out / "ProjectSettings").mkdir(exist_ok=True)
    (out / "ProjectSettings" / "ProjectVersion.txt").write_text(f"m_EditorVersion: {version}\n", encoding="utf-8")
    print(f"Wrote {out}: {copied} files copied, {len(docs)} documents rendered, {metas} .meta files generated"
          + (" (tests included: do not upload this variant)" if args.with_tests else ""))

    errors, warnings = validate(out)
    for line in warnings:
        print("note: " + line)
    for line in errors:
        print("BLOCKER: " + line)
    print(f"{len(errors)} submission blocker(s) found by the offline checks." if errors
          else "Offline checks passed.")

    unity_ok = True
    if args.verify:
        unity_ok = run_unity(out, editor_root, args.with_tests)
    sys.exit(1 if errors or not unity_ok else 0)


if __name__ == "__main__":
    main()
