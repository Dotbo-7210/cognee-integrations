"""Manifest and plugin-surface contract for the Cursor Cognee plugin.

Checks the pieces Cursor reads (``.cursor-plugin/plugin.json``,
``hooks/hooks.json``), the launcher, the installer and the host-specific
constants of the runtime copy, without importing the runtime (its modules read
HOME and the env file at import time).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

INTEGRATIONS_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = INTEGRATIONS_ROOT.parent
PLUGIN_ROOT = INTEGRATIONS_ROOT / "cursor"
PLUGIN_JSON = PLUGIN_ROOT / ".cursor-plugin" / "plugin.json"
HOOKS_JSON = PLUGIN_ROOT / "hooks" / "hooks.json"
SCRIPTS_DIR = PLUGIN_ROOT / "scripts"
ADAPTER = SCRIPTS_DIR / "cursor_hook.py"
INSTALLER = SCRIPTS_DIR / "install-cursor-hooks.py"
POSIX_LAUNCHER = SCRIPTS_DIR / "run-cursor-hook"
WINDOWS_LAUNCHER = SCRIPTS_DIR / "run-cursor-hook.cmd"
CODEX_SCRIPTS = INTEGRATIONS_ROOT / "codex" / "plugins" / "cognee" / "scripts"

#: Agent hook names Cursor documents (https://cursor.com/docs/hooks).
CURSOR_AGENT_HOOKS = {
    "sessionStart",
    "sessionEnd",
    "preToolUse",
    "postToolUse",
    "postToolUseFailure",
    "subagentStart",
    "subagentStop",
    "beforeShellExecution",
    "afterShellExecution",
    "beforeMCPExecution",
    "afterMCPExecution",
    "beforeReadFile",
    "afterFileEdit",
    "beforeSubmitPrompt",
    "preCompact",
    "stop",
    "afterAgentResponse",
    "afterAgentThought",
}


@pytest.fixture
def plugin_root() -> Path:
    if not PLUGIN_JSON.is_file():
        pytest.skip(f"Cursor plugin has not been implemented: {PLUGIN_ROOT}")
    return PLUGIN_ROOT


@pytest.fixture
def manifest(plugin_root: Path) -> dict:
    return json.loads(PLUGIN_JSON.read_text(encoding="utf-8"))


@pytest.fixture
def hooks(plugin_root: Path) -> dict:
    assert HOOKS_JSON.is_file(), f"missing Cursor hook manifest: {HOOKS_JSON}"
    return json.loads(HOOKS_JSON.read_text(encoding="utf-8"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(plugin_root: Path):
    module = _load(ADAPTER, "cursor_contract_adapter")
    try:
        yield module
    finally:
        sys.modules.pop("cursor_contract_adapter", None)


# --------------------------------------------------------------------------- #
# .cursor-plugin/plugin.json
# --------------------------------------------------------------------------- #


def test_plugin_manifest_is_a_valid_cursor_plugin(manifest):
    assert re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]", manifest["name"]), manifest["name"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"]), manifest["version"]
    assert manifest["description"].strip()
    assert manifest["author"]["name"]
    assert (PLUGIN_ROOT / manifest["hooks"]).resolve() == HOOKS_JSON.resolve()
    skills_dir = (PLUGIN_ROOT / manifest["skills"]).resolve()
    assert skills_dir.is_dir()
    for path in (manifest["hooks"], manifest["skills"]):
        assert not path.startswith("/") and ".." not in path, path


def test_root_marketplace_lists_the_cursor_plugin_like_the_other_hosts(manifest):
    """Cursor installs from a Git repo listed in a marketplace; a multi-plugin repo
    needs ``.cursor-plugin/marketplace.json`` at the root pointing at the plugin dir,
    mirroring ``.claude-plugin/marketplace.json`` for Claude Code."""
    marketplace_path = REPO_ROOT / ".cursor-plugin" / "marketplace.json"
    assert marketplace_path.is_file(), marketplace_path
    marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]", marketplace["name"])
    assert marketplace["owner"]["name"]
    entries = [
        entry
        for entry in marketplace["plugins"]
        if str(entry["source"]).removeprefix("./").rstrip("/") == "integrations/cursor"
    ]
    assert len(entries) == 1, marketplace["plugins"]
    entry = entries[0]
    assert entry["name"] == manifest["name"]
    assert entry["version"] == manifest["version"]
    assert entry["description"].strip()
    source = str(entry["source"])
    assert not source.startswith("/") and ".." not in source, source
    assert (
        REPO_ROOT / source / ".cursor-plugin" / "plugin.json"
    ).resolve() == PLUGIN_JSON.resolve()
    names = [entry["name"] for entry in marketplace["plugins"]]
    assert len(names) == len(set(names)), names


def test_every_skill_directory_has_a_skill_md_naming_cursor_not_codex(manifest):
    skills_dir = PLUGIN_ROOT / manifest["skills"]
    skills = sorted(p for p in skills_dir.iterdir() if p.is_dir())
    assert skills, "no skills shipped"
    for skill in skills:
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\n"), skill
        assert re.search(r"^name:\s*\S+", text, re.M), skill
        assert re.search(r"^description:\s*\S+", text, re.M), skill
        assert "CODEX_PLUGIN_ROOT" not in text, skill
        assert "COGNEE_CODEX_BACKEND" not in text, skill


def test_readme_and_changelog_exist_and_agree_on_the_version(manifest):
    assert (PLUGIN_ROOT / "README.md").is_file()
    changelog = (PLUGIN_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{manifest['version']}]" in changelog


# --------------------------------------------------------------------------- #
# hooks/hooks.json
# --------------------------------------------------------------------------- #


def test_hooks_json_uses_cursor_native_shape(hooks):
    assert hooks["version"] == 1
    assert isinstance(hooks["hooks"], dict) and hooks["hooks"]
    for event, entries in hooks["hooks"].items():
        assert event in CURSOR_AGENT_HOOKS, f"unknown Cursor hook {event!r}"
        assert isinstance(entries, list) and entries, event
        for entry in entries:
            assert set(entry) <= {"command", "timeout", "matcher", "failClosed", "type"}, entry
            assert isinstance(entry["command"], str)
            assert isinstance(entry["timeout"], int) and entry["timeout"] > 0
            assert "hooks" not in entry, "Claude Code nesting is not Cursor's format"


def _tokens(command: str) -> tuple[str, list[str]]:
    assert not any(token in command for token in ("$", "`", "~", "..", "||", ";", "&&")), command
    tokens = shlex.split(command)
    return tokens[0], tokens[1:]


def test_every_hook_runs_the_launcher_from_the_plugin_root(hooks, adapter):
    for event, entries in hooks["hooks"].items():
        for entry in entries:
            launcher, args = _tokens(entry["command"])
            assert launcher == "./scripts/run-cursor-hook", entry["command"]
            assert args and args[0] in adapter.EVENT_FOR_SCRIPT, entry["command"]
            assert all(flag.startswith("--") for flag in args[1:]), entry["command"]
            if args[0] != adapter.CACHE_RESPONSE:
                assert (SCRIPTS_DIR / args[0]).is_file(), args[0]


def test_hooks_json_matches_the_adapter_hook_table(hooks, adapter):
    rendered = {
        event: [
            (shlex.split(e["command"])[1], tuple(shlex.split(e["command"])[2:]), e["timeout"])
            for e in entries
        ]
        for event, entries in hooks["hooks"].items()
    }
    assert rendered == {event: list(entries) for event, entries in adapter.HOOK_TABLE.items()}


def test_the_expected_lifecycle_is_covered(hooks):
    events = set(hooks["hooks"])
    assert {"sessionStart", "beforeSubmitPrompt", "postToolUse", "stop", "sessionEnd"} <= events
    assert "afterAgentResponse" in events, "stop has no assistant text; the answer comes from here"
    assert "preToolUse" not in events, "a memory plugin must not gate tool calls"
    assert "beforeShellExecution" not in events
    assert "beforeMCPExecution" not in events
    stop_scripts = [shlex.split(e["command"])[1:] for e in hooks["hooks"]["stop"]]
    assert ["store-to-session.py", "--stop"] in stop_scripts
    end_scripts = [shlex.split(e["command"])[1:] for e in hooks["hooks"]["sessionEnd"]]
    assert ["sync-session-to-graph.py", "--session-end"] in end_scripts


# --------------------------------------------------------------------------- #
# launchers and installer
# --------------------------------------------------------------------------- #


def test_launchers_exist_and_the_posix_one_is_executable(plugin_root):
    assert POSIX_LAUNCHER.is_file() and WINDOWS_LAUNCHER.is_file()
    text = POSIX_LAUNCHER.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh"), "must run without bash"
    assert 'cursor_hook.py" "$@"' in text
    assert "python3" in text and "python " in text, "python3 first, python fallback"
    if os.name != "nt":
        assert POSIX_LAUNCHER.stat().st_mode & stat.S_IXUSR, "run-cursor-hook must be executable"
    assert "cursor_hook.py" in WINDOWS_LAUNCHER.read_text(encoding="utf-8")


def test_posix_launcher_replies_neutrally_when_stdin_is_empty(plugin_root, tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX launcher")
    proc = subprocess.run(
        [str(POSIX_LAUNCHER), "store-to-session.py", "--stop"],
        input="",
        text=True,
        capture_output=True,
        env={**os.environ, "HOME": str(tmp_path)},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip()) == {}


def test_installer_merges_and_uninstalls_without_touching_foreign_hooks(plugin_root, adapter):
    installer = _load(INSTALLER, "cursor_contract_installer")
    try:
        foreign = {"command": "./hooks/audit.sh", "timeout": 5}
        existing = {"version": 1, "hooks": {"stop": [foreign], "afterFileEdit": [foreign]}}
        merged = installer.merge(existing)
        assert merged["version"] == 1
        assert merged["hooks"]["afterFileEdit"] == [foreign]
        assert merged["hooks"]["stop"][0] == foreign
        ours = [e for e in merged["hooks"]["stop"] if "run-cursor-hook" in e["command"]]
        assert len(ours) == len(adapter.HOOK_TABLE["stop"])
        for entry in ours:
            tokens = shlex.split(entry["command"], posix=os.name != "nt")
            assert Path(tokens[0].strip('"')).resolve() == installer.launcher_path().resolve()
        assert set(merged["hooks"]) == set(adapter.HOOK_TABLE) | {"afterFileEdit"}

        again = installer.merge(merged)
        assert again == merged, "installing twice must not duplicate entries"

        removed = installer.merge(merged, uninstall=True)
        assert removed == {"version": 1, "hooks": {"stop": [foreign], "afterFileEdit": [foreign]}}
        assert installer.merge({}, uninstall=True) == {"version": 1, "hooks": {}}
    finally:
        sys.modules.pop("cursor_contract_installer", None)


def test_installer_cli_writes_a_project_hooks_json(plugin_root, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    proc = subprocess.run(
        [sys.executable, str(INSTALLER), "--project", str(project), "--print"],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["version"] == 1
    assert not (project / ".cursor").exists(), "--print must not write"

    proc = subprocess.run(
        [sys.executable, str(INSTALLER), "--project", str(project)],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    written = json.loads((project / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
    assert "beforeSubmitPrompt" in written["hooks"]
    assert not (tmp_path / "home" / ".cursor").exists(), "--project must not touch ~/.cursor"

    proc = subprocess.run(
        [sys.executable, str(INSTALLER), "--project", str(project), "--uninstall"],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        json.loads((project / ".cursor" / "hooks.json").read_text(encoding="utf-8"))["hooks"] == {}
    )


# --------------------------------------------------------------------------- #
# the runtime copy
# --------------------------------------------------------------------------- #


def _source(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def test_runtime_constants_are_cursors(plugin_root):
    config = _source("config.py")
    assert '_STATE_DIR = Path.home() / ".cognee-plugin" / "cursor"' in config
    assert '"agent_name": "cursor-agent"' in config
    assert '"session_prefix": "cursor"' in config
    assert '"COGNEE_CURSOR_BACKEND": "backend"' in config
    assert 'os.environ.get("CURSOR_CWD"' in config
    common = _source("_plugin_common.py")
    assert 'PLUGIN_KEY = "cursor"' in common
    assert 'CONNECTION_TYPE = "cursor"' in common
    assert '_PLUGIN_DIR = Path.home() / ".cognee-plugin" / "cursor"' in common
    assert '_UPDATE_MANIFEST_PATH = "integrations/cursor/.cursor-plugin/plugin.json"' in common
    assert '_STATE_SUBDIR = "cursor"' in _source("hook_runner.py")
    assert '_INTEGRATION = "cursor"' in _source("_code_graph.py")
    assert '_PLUGIN_BACKEND_VAR = "COGNEE_CURSOR_BACKEND"' in _source("_env_file.py")
    assert "_find_cursor_parent_pid" in _source("session-start.py")


def test_no_codex_state_or_manifest_paths_survive_in_the_copy(plugin_root):
    offenders = []
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in (
            '".cognee-plugin" / "codex"',
            '_SHARED_ROOT / "codex"',
            ".codex-plugin",
            "CODEX_CWD",
            "COGNEE_CODEX_BACKEND",
            "codex-agent",
        ):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, offenders


def test_runtime_copy_tracks_the_codex_runtime(plugin_root):
    """Every codex script exists here; only the host-specific ones may differ."""
    if not CODEX_SCRIPTS.is_dir():
        pytest.skip("codex runtime not present")
    missing = [
        p.name
        for p in CODEX_SCRIPTS.iterdir()
        if p.is_file() and not (SCRIPTS_DIR / p.name).exists()
    ]
    assert not missing, f"cursor copy lacks codex scripts: {missing}"
    cursor_only = {
        "cursor_hook.py",
        "run-cursor-hook",
        "run-cursor-hook.cmd",
        "install-cursor-hooks.py",
    }
    extra = {p.name for p in SCRIPTS_DIR.iterdir() if p.is_file()} - {
        p.name for p in CODEX_SCRIPTS.iterdir()
    }
    assert extra == cursor_only, extra


def test_runtime_scripts_parse_on_python_3_9_syntax(plugin_root):
    """Hooks run under the oldest python3 on PATH (macOS ships 3.9.6)."""
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
        except SyntaxError as exc:
            pytest.fail(f"{path.name} is not Python 3.9 syntax: {exc}")
        # ``X | None`` annotations need the future import to survive 3.9.
        if "| None" in source or re.search(r":\s*\w+\s*\|\s*\w+", source):
            has_future = any(
                isinstance(node, ast.ImportFrom) and node.module == "__future__"
                for node in tree.body
            )
            assert has_future, (
                f"{path.name} uses PEP 604 unions without `from __future__ import annotations`"
            )


def test_shell_helpers_are_executable(plugin_root):
    if os.name == "nt":
        pytest.skip("POSIX bits")
    for path in sorted(SCRIPTS_DIR.glob("*.sh")) + [SCRIPTS_DIR / "cognee-plugin", POSIX_LAUNCHER]:
        assert path.stat().st_mode & stat.S_IXUSR, f"{path.name} lost its executable bit"
