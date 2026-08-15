---
name: sonar-quality
description: Prevent SonarCloud/SonarQube issues when writing or changing code in this repo.
---

# Anaglyph Sonar Quality Standards

Apply these rules when writing or changing code in this repo. The repository's Sonar
project is `bckirkup_Anaglyph`, with analysis wired through
`.github/workflows/ci.yml`.

## Local validation

```bash
pre-commit run --all-files
python scripts/sonar_guard.py main.py camera_manager.py calibration.py gui.py compositor.py video_recorder.py tests
python scripts/sonar_guard.py --workflows .github/workflows
uv run --no-sync --no-build ruff check main.py camera_manager.py calibration.py gui.py compositor.py video_recorder.py tests/
uv run --no-sync --no-build ruff format --check main.py camera_manager.py calibration.py gui.py compositor.py video_recorder.py tests/
uv run --no-sync --no-build mypy main.py camera_manager.py calibration.py gui.py --ignore-missing-imports || true
uv run --no-sync --no-build pytest tests/ -v -m "not hardware" --tb=short
```

## Rule catalog

- `python:S3776`: keep cognitive complexity at or below the repository's current
  complexity ceiling. New or changed functions must stay below the pre-commit
  threshold of 15.
- `python:S9073`: use separate assertions for separate conditions; do not write
  `assert condition_a and condition_b`.
- `python:S1172`: rename intentionally unused parameters with a leading underscore.
- `python:S116` and `python:S117`: use conventional snake_case names for Python
  methods, variables, and attributes.
- `python:S107`: keep function signatures focused; group related options into a
  configuration object when a signature would otherwise grow too large.
- `githubactions:S8541`: published-package `pip install` commands must include
  `--only-binary :all:`.
- `githubactions:S8544`: published-package installs must use explicit versions,
  immutable commit references, or hashes. Local editable installs such as
  `pip install -e .` are exempt because their version is defined by the checked-out
  project metadata; they must still be reviewed for dependency reproducibility.
- The same workflow checks cover UV installers:
  - `uv sync` requires `--no-build` and `--locked` or `--frozen`.
  - `uv pip install`, `uv add`, and `uv tool install` require `--no-build` and
    pinned or hashed requirements.
  - `uvx` and `uv tool run` require `--no-build` and a pinned `--from
    package==version` requirement.
  - `--no-binary-package <package>` may document source-build exceptions, but
    does not replace `--no-build`.
  - `uv run` requires `--no-build`, including when combined with `--no-sync`.
  - Non-installing commands such as `uv lock` and `uv --version` are not checked.

The mechanical guard is intentionally conservative and checks Python files plus
workflow YAML. `zizmor` separately checks GitHub Actions action pinning.

## CI complexity ratchet

CI checks the whole repository with the current `ruff` complexity ceiling recorded
in `pyproject.toml`. That ceiling only ratchets downward. Pre-commit applies the
goal threshold of 15 to changed Python files.

## Existing source findings

Do not add `# noqa`, `nosonar`, or equivalent suppressions. Fix unused parameters,
naming, complexity, and oversized signatures at their causes. Taint-analysis
findings require review in the SonarQube Cloud job because local Ruff and the
mechanical guard cannot prove inter-procedural data flow.

## Validating the guards before copying this template to another repo

After copying this template or editing `scripts/sonar_guard.py`,
`.pre-commit-config.yaml`, or `.github/workflows/ci.yml`, prove that the guards reject
bad code and stay quiet on good code. Plant one defect at a time and run it through
the real hook path; a direct script invocation alone does not prove the hook is wired.
Use a scratch branch and never push it:

```bash
git checkout -b scratch/guard-validation origin/main
# plant one defect, then:
git add <file> && git commit -m scratch   # must be refused, naming the rule ID
git reset --hard origin/main              # discard; never push a scratch branch
```

The workflow guard checks `run:` installer commands, while zizmor checks `uses:`
action pinning. Both hooks are required; dropping either leaves a whole class
unguarded.

### Generating workflow probes without fooling yourself

Do not build probe YAML with `sed` or hand-written quoting. A `run:` value containing
`:all:` or nested quotes can produce invalid YAML, causing zizmor to report
`failed to parse input` and making the guard tokenize a mangled command. Emit probe
fixtures with a YAML dumper and assert that the parsed `run` string equals the
intended command before trusting the result.

### Known guard blind spots

The Python rules are deliberately conservative, so a clean local guard is necessary
but not sufficient:

- `assert value == _expected_threshold()` remains a SonarCloud-side catch. The guard
  does not infer helper return types or perform inter-procedural type analysis.
- Runtime types are not inferred across attributes, subscripts, module scope, or
  function boundaries. A simple local float binding such as
  `expected = 0.5; assert value == expected` is covered only when that function has
  one unambiguous float assignment to the name; conflicting or dynamic assignments
  stay silent.
- `S9073` is only applied to test files (`tests/` in the path, `test_*.py`, or
  `*_test.py`). Nested `and` expressions inside an assertion are reported once per
  assertion.

Rely on the `SonarQube Cloud` CI job for these blind spots and for findings requiring
type or data-flow inference.

### UV command spelling

`uvx --from pkg==version` without `--no-build` is correctly reported as `S8541`.
The compliant form is:

```text
uvx --no-build --from pkg==version ...
```
