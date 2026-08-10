# agent-project


- QwenVL-3B model. Goal: build a GUI agent under a constrained environment (16GB RAM).
- The QwenVL-3B model has separate qLoRA fine-tunes for planning and grounding.
- To compensate for weak grounding, the system from the RegionFocus paper is used.

## Architecture

### 1. Model — single instance, LoRA adapter switching (`agent_loop.py`)

Running separate model instances for planning and grounding would exceed the 16GB RAM budget (CPU-only, no CUDA), so the project loads **one QwenVL-3B instance** with two named LoRA adapters attached at once (`default` = grounding, `planner`), and switches between them at runtime via `peft`'s `set_adapter()` / `disable_adapter()`.

- `_AdapterSwitchView`: switches to the `planner` adapter only for the duration of a planning call, then restores `default` (grounding) afterward — restoration is guaranteed via `finally` even if generation raises.
- `_BaseModelView`: for setups without a planner LoRA, planning instead runs under `disable_adapter()` (pure base model behavior).
- Both LoRAs only train `target_modules=["q_proj","k_proj","v_proj","o_proj"]`, so the vision encoder is identical regardless of which adapter is active — "how well the model sees the screen" never changes; only the output format (coordinates vs. JSON action) does.

### 2. Planner (`planner.py`)

Prompts the base/planner-LoRA model in a ReAct style to decide **a single next action** as JSON. It never outputs coordinates — click-type actions only produce a natural-language `target_description`, deferring the actual coordinate lookup to grounding.

```json
{"reasoning": "...", "action": "left_click|...|terminate",
 "target_description": "...", "text": "...", "status": "...", "answer": "..."}
```

**Pre-execution review loop (`plan_with_reflection`)**: a separate critic call reviews the candidate action before it is ever executed.
- The reflector always re-observes the screenshot independently first, rather than taking the proposer's stated reasoning at face value.
- If rejected, the critique is fed back into the planner for a revised attempt (default `max_iterations=2`).
- If still rejected after all iterations, the action is not executed — but the rejection itself is recorded in history via `_rejected` / `_rejection_reason`, so the next step's planner knows the previous attempt failed and doesn't just repeat it.

### 3. Grounding — RegionFocus (`region_focus.py`)

Converts a `target_description` into pixel coordinates using a local reproduction of the RegionFocus paper's pipeline:

1. Initial grounding (one coordinate inference pass)
2. Judge — the model itself decides YES/NO on whether the predicted point is correct
3. If wrong, `region_focus()` retries at increasing temperatures to find a new candidate
4. Crop + upsample around the candidate at 4 different zoom ratios and re-infer more precisely (Step 4)
5. If multiple candidates remain, have the model pick the best one (aggregation)

Coordinates are queried using the exact format the grounding LoRA was fine-tuned on (0–1000 normalized text, `(x,y)`) — using a tool-call schema instead measurably hurt accuracy, since the LoRA never saw that format during training.

### 4. Execution environment (`env_webvoyager.py`)

A Selenium-based wrapper targeting the WebVoyager benchmark (live websites). It only handles `reset(task) → screenshot` and `execute_action(action) → screenshot` — it does not compute reward or success (that's the evaluation layer's job, via an LLM judge).

- Coordinate clicks/scrolls use **CDP (`Input.dispatchMouseEvent`, etc.)** directly instead of Selenium ActionChains, to guarantee the same coordinate space as the screenshot.
- Includes headless-automation-detection workarounds (spoofed user agent, removing `navigator.webdriver`, etc.).

### 5. Wiring the policy together — Planner + Grounding + Env (`eval_webvoyager.py`)

`build_planner_grounding_agent_step()` combines the pieces above into a single policy function:

```
screenshot → plan_with_reflection() [planner LoRA, retried/rejected until approved]
           → target_description resolved to coordinates via ground_with_regionfocus() [grounding LoRA]
           → env.execute_action(coordinate-based action)
```

- The reflection/critic call always runs with the planner LoRA disabled (base model), since the planner LoRA has never seen the critic's expected output format.
- If a `terminate` action has no final answer filled in, a separate QA call extracts one (the planner LoRA's training data never populated this field, so it can't be fixed by retraining alone).
- When an action gets silently downgraded to a no-op (grounding failure, unimplemented drag, etc.), that fact is explicitly recorded in history too — so the next planning step knows the previous attempt didn't actually happen and doesn't just retry blindly.

### 6. Evaluation (`eval_webvoyager.py` — `run_episode` / `run_batch`)

Each task runs for up to `max_steps` (default 15). The last 15 screenshots are shown to an LLM judge (local Qwen or OpenAI GPT-4o, selectable) to determine success. The judge isn't trusted on a single call — it runs 3 times by default and the final verdict is decided by majority vote.

### 7. Testing strategy

Every module (except `region_focus.py`) has a mock-based `--selftest` that verifies logic without needing a real model or browser. Model calls are replaced with `MagicMock`, so prompt assembly, parsing, and state handling (e.g. how a rejected action gets recorded in history) can be checked quickly — actual accuracy is still verified separately in an environment with a real GPU/browser.