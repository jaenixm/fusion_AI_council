# fusion-AI-council

`fusion-AI-council` is a local, CLI-first model council for questions that benefit from several independent analyses and a final synthesis.

For each task it:

1. Builds an optional, size-limited context bundle from files you select.
2. Runs Gemini, Claude, and Codex in parallel.
3. Saves every worker response.
4. Removes model identities, randomizes the successful responses, and sends them to a judge model.
5. Extracts and prints one final answer while preserving the full audit trail locally.

This is useful for:

- Coding reviews, debugging, architecture, and implementation planning.
- Scientific and thesis writing grounded in local source material.
- Decisions where independent reasoning and contradiction resolution are valuable.
- Reproducible comparison of model outputs.

## How it works

```text
Question + optional context
             |
             v
   Mode-specific prompt
             |
     +-------+-------+
     |       |       |
     v       v       v
  Gemini   Claude   Codex
     |       |       |
     +-------+-------+
             |
   successful responses
   shuffled and anonymized
             |
             v
       Judge model
             |
             v
   Final synthesized answer
```

The three initial workers run concurrently. A failed worker is recorded in the run directory but is not shown to the judge. If every worker fails, judging is skipped and the run ends with an error result.

The default council is:

| Council member | Backend | Default model |
|---|---|---|
| Gemini | `agy` | `Gemini 3.1 Pro (High)` |
| Claude | native `claude` CLI | `opus`, effort `xhigh` |
| Codex | `codex exec` | `gpt-5.5`, reasoning `xhigh` |

Codex is the default judge. The judge can instead be Claude or Gemini.

## Requirements

- Python 3.11 or newer.
- The `agy`, `claude`, and `codex` commands installed, authenticated, and available on `PATH`.
- Access to the models configured for those CLIs.

Only PyYAML is required as a Python runtime dependency.

Check the external commands before running a council:

```bash
agy --help
claude --version
codex --version
```

The project does not manage authentication or API keys. Configure each external CLI according to its own documentation, and keep credentials outside this repository.

## Installation

Using `uv`:

```bash
uv sync
uv run fusion-council "Your question"
```

Using a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
fusion-council "Your question"
```

The repository-level script is also supported:

```bash
python3 fusion.py "Your question"
```

## Basic usage

Ask a general question:

```bash
fusion-council "Compare these two implementation approaches."
```

Let the model CLIs inspect another project from its working directory:

```bash
fusion-council \
  --workdir /path/to/project \
  --mode coding \
  "Review the architecture and identify concrete failure modes."
```

Add explicit context:

```bash
fusion-council \
  --workdir /path/to/project \
  --mode coding \
  --context README.md \
  --context docs/architecture.md \
  "Create an implementation plan."
```

The final answer is printed to the terminal and the complete run is saved under `runs/`.

## Task modes

The `--mode` option changes context limits and prompting:

| Mode | Intended use | Total context budget | Per-file limit |
|---|---|---:|---:|
| `general` | General reasoning and decisions | 150,000 characters | 60,000 |
| `coding` | Engineering, code review, and debugging | 200,000 characters | 100,000 |
| `scientific-writing` | Thesis and literature-grounded writing | 600,000 characters | 180,000 |

`scientific-writing` tells workers and the judge to use supplied material as the primary evidence base and not invent citations, paper claims, or bibliography details.

## Context collection

Context is explicit rather than a full automatic repository dump. Pass `--context` multiple times to include files or directories:

```bash
fusion-council \
  --workdir /path/to/project \
  --context README.md \
  --context docs/ \
  "Explain how this project works."
```

Relative paths are resolved against `--workdir`. Supported text formats are:

```text
.md .txt .bib .tex .rst .json .yaml .yml
```

When a directory is supplied, supported files are collected recursively in sorted order. The collector skips:

```text
.git .venv __pycache__ .pytest_cache checkpoints node_modules vendors
```

### Local link following

By default, the collector also includes supported local files linked from directly selected Markdown files:

```markdown
[Architecture](docs/architecture.md)
[Research notes](file:///absolute/path/notes.md)
```

HTTP links, HTTPS links, and page anchors are ignored. Link discovery is one level deep; links inside newly discovered files are not recursively followed.

Disable this behavior with:

```bash
fusion-council --no-follow-links --context README.md "Review this."
```

### Scientific-writing auto-context

In `scientific-writing` mode, these files are included automatically when they exist under `--workdir`:

```text
context/llm/llm_thesis_context_compressed.md
context/literature/_index.md
context/progress/thesis_progress.md
```

Disable automatic additions with `--no-auto-context`.

### Priority, deduplication, and limits

Context sources are deduplicated by resolved path and processed in this order:

1. Explicit files.
2. Files discovered inside explicit directories and files linked from Markdown.
3. Automatic scientific-writing files.

The collector takes at most the per-file limit from each source until the total mode budget is exhausted. Both limits can be overridden:

```bash
fusion-council \
  --context docs/ \
  --context-budget-chars 50000 \
  --max-context-file-chars 10000 \
  "Analyze this material."
```

Each source is placed inside a random delimiter and labeled with its path and inclusion reason. Worker and judge prompts explicitly treat the bundle as untrusted reference material rather than instructions.

### Preview context without calling models

Use a dry run to inspect exactly what would be included:

```bash
fusion-council \
  --workdir /path/to/project \
  --context docs/ \
  --dry-run-context \
  "Context preview"
```

This writes `00_context_bundle.md`, prints the source manifest, and exits without invoking any model.

Context collection and model workspace access are separate: the bundle is a fixed text snapshot, while agentic model CLIs may inspect additional files available in `--workdir`.

## Run artifacts

Each normal run is stored in a timestamped directory:

```text
runs/YYYYMMDD_HHMMSS/
  00_context_bundle.md   # omitted when no context was collected
  01_gemini.md
  02_claude.md
  03_codex_initial.md
  04_judge_raw.md
  04_final.md
  metadata.json
```

`metadata.json` records:

- The original question and task mode.
- Project working directory.
- Context paths, inclusion reasons, character counts, and truncation.
- Worker configuration and success status.
- Randomized judge input order.
- Judge backend and model.
- Final extraction status.

Run artifacts may contain source material, absolute paths, prompts, model responses, and other sensitive project information. They are ignored by Git and should be treated as private local data.

## Rerunning part of a council

An existing run can be updated without repeating every worker:

```bash
fusion-council \
  --rerun-run runs/20260624_111152 \
  --rerun-model claude \
  --rejudge
```

Available worker keys are `gemini`, `claude`, and `codex`.

Judge again without rerunning a worker:

```bash
fusion-council \
  --rerun-run runs/20260624_111152 \
  --rejudge
```

Replaced worker and judge files are backed up with a `.bak_TIMESTAMP` suffix. Reruns use the saved context bundle; they do not recollect changed source files.

## Configuration

Configuration precedence is:

```text
built-in defaults < YAML file < environment variables < CLI options
```

Example YAML:

```yaml
judge: claude

models:
  gemini:
    backend: agy
    model: Gemini 3.1 Pro (High)
    label: Gemini 3.1 Pro High

  claude:
    backend: claude
    model: opus
    reasoning_effort: xhigh
    label: Claude Opus xhigh

  codex:
    backend: codex
    model: gpt-5.5
    reasoning_effort: xhigh
    label: Codex GPT-5.5 xhigh

context:
  budget_chars:
    general: 150000
    coding: 200000
    scientific-writing: 600000
  max_file_chars:
    general: 60000
    coding: 100000
    scientific-writing: 180000
```

Run with:

```bash
fusion-council --config fusion-council.yaml "Your question"
```

Supported environment overrides:

```text
FUSION_JUDGE
FUSION_CODEX_MODEL
FUSION_CODEX_REASONING_EFFORT
FUSION_CLAUDE_MODEL
FUSION_CLAUDE_EFFORT
FUSION_CLAUDE_REASONING_EFFORT
FUSION_CONTEXT_BUDGET_CHARS
FUSION_MAX_CONTEXT_FILE_CHARS
```

Common CLI overrides:

```bash
fusion-council \
  --judge claude \
  --codex-model gpt-5.5 \
  --codex-reasoning high \
  --claude-model opus \
  --claude-effort max \
  "Your question"
```

Run `fusion-council --help` for the complete option list.

## Development

Run the tests through the active Python interpreter:

```bash
python3 -m pytest -q
```

The test suite covers configuration precedence, context discovery and truncation, prompt construction, model command generation, artifact handling, failure exclusion, output persistence, and partial reruns.

## Security and privacy

- Do not put API keys or credentials in prompts, context files, YAML configuration, or run artifacts.
- Context paths may be absolute and may point outside `--workdir`; verify them with `--dry-run-context`.
- External model CLIs run with access permitted by their own configuration and with `--workdir` as their current directory.
- The council stores model responses and context locally in plain text.
- `.gitignore` excludes runs, local environments, common credential files, private/local configuration, caches, logs, and model-tool state.
- If sensitive files were committed before `.gitignore` was added, ignoring them does not remove them from Git history. Rotate exposed credentials and clean the repository history separately.

## Current limitations

- Council membership is currently fixed to Gemini, Claude, and Codex.
- Context link following is only one level deep.
- General and coding modes do not automatically discover repository files.
- Timestamped run directories have one-second precision, so truly simultaneous launches may collide.
- A context-free historical run cannot currently be rerun unless it has a saved context bundle.
- The judge output contract is prompt-based; if the judge omits `<final_answer>`, its complete response is used as the final answer.
