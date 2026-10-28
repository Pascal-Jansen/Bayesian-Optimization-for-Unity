"""Files the release carries in several places agree: the version (the release workflow bumps them together) and
the license and notice copies in the separately installed parts."""

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def single(pattern, relative):
    matches = re.findall(pattern, read(relative), re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"{relative}: expected one match of {pattern!r}, found {len(matches)}")
    return matches[0]


class VersionTests(unittest.TestCase):
    def test_version_files_agree(self):
        version = read("VERSION").strip()
        python_version = re.sub(r"-(alpha|beta|rc)\.", lambda m: {"alpha": "a", "beta": "b", "rc": "rc"}[m.group(1)], version)
        plugin = json.loads(read("Unreal/Plugins/BayesianOptimization/BayesianOptimization.uplugin"))
        self.assertEqual(single(r'^version: "([^"]+)"$', "CITATION.cff"), version)
        self.assertEqual(single(r'^version = "([^"]+)"$', "Python/pyproject.toml"), python_version)
        self.assertEqual(single(r'^__version__ = "([^"]+)"$', "Python/interactive_bo/__init__.py"), python_version)
        self.assertEqual(single(r"^  bundleVersion: (.+)$", "Unity/ProjectSettings/ProjectSettings.asset"), version)
        self.assertEqual(plugin["VersionName"], version)
        self.assertEqual(plugin["IsBetaVersion"], "-" in version)

    def test_license_copies_match(self):
        for copy in ("Backend/LICENSE", "Python/LICENSE", "Unreal/Plugins/BayesianOptimization/LICENSE",
                     "Unity/Assets/iBO/Documentation/LICENSE.txt"):
            with self.subTest(copy=copy):
                self.assertEqual(read(copy), read("LICENSE"))

    def test_third_party_notice_copies_match(self):
        for copy in ("Unreal/THIRD-PARTY-NOTICES.txt", "Unity/Assets/iBO/Documentation/Third-Party Notices.txt"):
            with self.subTest(copy=copy):
                self.assertEqual(read(copy), read("THIRD-PARTY-NOTICES.txt"))


if __name__ == "__main__":
    unittest.main()
