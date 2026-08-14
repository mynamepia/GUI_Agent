# agent-project

- Goal: build a GUI agent that can browse and operate real websites (WebVoyager benchmark), under a 16GB VRAM budget (RTX 5070ti + 16GB system shared memory as an OOM buffer).
- The project supports **two interchangeable pipelines**, selectable per run:
  - **Track A — local LoRA** (`agent/`): a single QwenVL-3B instance with separate qLoRA adapters for planning and grounding.
  - **Track B — GUI-Actor + API planner** (`agent-actor3b/`): grounding via the pretrained `microsoft/GUI-Actor-3B-Qwen2.5-VL` (coordinate-free pointer model), planning via an OpenAI vision model (GPT-4o).
- Both tracks share the same execution/evaluation layer (`env_webvoyager.py`, `planner.py`).

## Track A — local LoRA pipeline (`agent/`)

### 1. Model — single instance, LoRA adapter switching (`agent_loop.py`)

Loads **one QwenVL-3B instance** with two named LoRA adapters attached at once (`default` = grounding, `planner`), switching between them at runtime via `peft`'s `set_adapter()` / `disable_adapter()` instead of running two separate model instances.

- `_AdapterSwitchView`: switches to the `planner` adapter for a planning call, restores `default` afterward.
- `_BaseModelView`: runs a call with adapters disabled (pure base model) — used for reflection/critic calls and final-answer QA, since those output formats were never seen during LoRA training.

### 2. Planner (`planner.py`)

Prompts the model (local LoRA, or an OpenAI vision model via `api_planner.py`) in a ReAct style to decide **a single next action** as JSON. Click-type actions produce a natural-language `target_description` only, not coordinates — grounding resolves that separately.

```json
{"reasoning": "...", "action": "left_click|...|terminate",
 "target_description": "...", "text": "...", "status": "...", "answer": "..."}
```

- `plan_with_reflection`: a critic call reviews each candidate action before execution and can reject it with feedback for a revised attempt.
- History formatting includes repetition detection (`REPETITION WARNING`) so the planner is nudged away from actions that aren't making progress. `scroll` is excluded from this check.

### 3. Grounding — RegionFocus (`region_focus.py`)

Converts a `target_description` into pixel coordinates: initial grounding → self-judge (YES/NO) → retry at higher temperature if rejected → crop/zoom re-inference → aggregation if multiple candidates remain. Coordinates are queried in the LoRA's fine-tuned text format (0–1000 normalized `(x,y)`).

## Track B — GUI-Actor pipeline (`agent-actor3b/`)

### `gui_actor_grounding.py` / `gui_actor_region_focus.py`

Wraps `microsoft/GUI-Actor-3B-Qwen2.5-VL`, a pretrained coordinate-free pointer model that attends over image patches and returns click points directly (no text coordinates). `gui_actor_region_focus.py` reimplements the same RegionFocus pipeline as Track A on top of GUI-Actor's output.

### `gui_actor_eval_webvoyager.py`

`build_planner_grounding_agent_step()` wires planner + grounding + env into one policy function. `--grounding_backend {lora, gui_actor}` and `--planner_backend {local, openai}` select which track's components are used. Also implements `drag` (start/end description each grounded separately, executed as a `left_click_drag` env action) and final-answer extraction for `terminate` actions missing an `answer` field (routed through the planner model, not the grounding model).

## Execution environment (`env_webvoyager.py`)

Selenium-based wrapper for live WebVoyager sites, shared by both tracks. `reset(task) → screenshot` / `execute_action(action) → screenshot` only — no reward/success logic (that's the evaluation layer's job).

- Clicks/scrolls/drags dispatched via CDP (`Input.dispatchMouseEvent`) instead of Selenium ActionChains.
- Automation-detection workarounds (spoofed user agent, `navigator.webdriver` removal) and CAPTCHA/bot-check detection with optional manual-solve support.
- Per-site locale forcing (cookies for Amazon/Apple, URL params for Google/Booking) so non-US IPs don't get served the wrong language/currency.
- New-tab handling (site opens content in a new tab → focus follows it) and a page-load wait before each screenshot.

## Evaluation (`eval_webvoyager.py` / `gui_actor_eval_webvoyager.py`)

`run_episode` / `run_batch`: each task runs up to `max_steps` (default 17), then the last few screenshots go to an LLM judge (local Qwen or GPT-4o) for a success verdict, optionally by majority vote over multiple judge calls. A windowed repeat-action detector ends an episode early if it's clearly stuck (excluding `scroll`). `eval_webvoyager.py` (Track A) also supports `--resume` (skip already-completed task IDs from a prior `--out` file) for long unattended runs.

## Testing strategy

Every module has a mock-based `--selftest` (no real model/browser needed) covering prompt assembly, parsing, and state handling. Actual accuracy is verified separately with a real GPU/browser.
