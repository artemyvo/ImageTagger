# Ollama Settings Reference

Navigation: [Project README](../README.md) | [Docs Index](README.md) | [Usage Guide](usage.md)

This page explains ImageTagger Ollama settings with a focus on auto thread mode.

For fixup regeneration troubleshooting, the merge dialog also provides local server/model controls and local Description/Tags prompt overrides. These are useful when the current model struggles to regenerate quality tags or descriptions with default settings.

## Scope

These settings affect Generate, Validate, and AI Find when `llm_threads` is set to `0` (auto mode).

If `llm_threads` is greater than `0`, fixed thread mode is used and auto thresholds are ignored.

## Core Settings

- `llm_endpoint`: Ollama server URL.
- `llm_model`: active model name.
- `llm_max_resolution_mpx`: input image downscale target in megapixels.
- `llm_threads`: thread count. Use `0` for auto mode.

## Auto Mode Thresholds

- `llm_auto_max_threads`: hard upper bound for auto mode.
- `llm_auto_window_size`: completions per control window.
- `llm_auto_warmup_items`: minimum completions before scale-up.
- `llm_auto_increase_retry_ratio_max`: maximum retry ratio allowed for scale-up.
- `llm_auto_increase_timeout_ratio_max`: maximum timeout ratio allowed for scale-up.
- `llm_auto_increase_latency_ratio_max`: maximum latency inflation allowed for scale-up.
- `llm_auto_decrease_retry_ratio_min`: retry ratio threshold to scale down.
- `llm_auto_decrease_timeout_ratio_min`: timeout ratio threshold for aggressive scale-down.
- `llm_auto_decrease_latency_ratio_min`: latency inflation threshold to scale down.
- `llm_auto_stable_throughput_factor`: required throughput floor vs previous window for scale-up.
- `llm_auto_healthy_streak_required`: healthy windows required before increasing threads.
- `llm_auto_cooldown_windows`: windows to wait after scale-down before increasing again.

## Presets

Use these as starting points. Keep `llm_threads` at `0` for all presets.

### Safe Preset

Best for mixed workloads, unstable servers, or conservative operation.

```json
{
  "llm_threads": 0,
  "llm_auto_max_threads": 16,
  "llm_auto_window_size": 10,
  "llm_auto_warmup_items": 10,
  "llm_auto_increase_retry_ratio_max": 0.03,
  "llm_auto_increase_timeout_ratio_max": 0.0,
  "llm_auto_increase_latency_ratio_max": 1.15,
  "llm_auto_decrease_retry_ratio_min": 0.10,
  "llm_auto_decrease_timeout_ratio_min": 0.0,
  "llm_auto_decrease_latency_ratio_min": 1.40,
  "llm_auto_stable_throughput_factor": 1.00,
  "llm_auto_healthy_streak_required": 2,
  "llm_auto_cooldown_windows": 1
}
```

### Balanced Preset

Current default behavior in the app.

```json
{
  "llm_threads": 0,
  "llm_auto_max_threads": 48,
  "llm_auto_window_size": 6,
  "llm_auto_warmup_items": 4,
  "llm_auto_increase_retry_ratio_max": 0.18,
  "llm_auto_increase_timeout_ratio_max": 0.0,
  "llm_auto_increase_latency_ratio_max": 1.35,
  "llm_auto_decrease_retry_ratio_min": 0.30,
  "llm_auto_decrease_timeout_ratio_min": 0.0,
  "llm_auto_decrease_latency_ratio_min": 2.0,
  "llm_auto_stable_throughput_factor": 0.90,
  "llm_auto_healthy_streak_required": 1,
  "llm_auto_cooldown_windows": 0
}
```

### Aggressive Preset

For fast GPUs and models where you want faster ramp-up and accept more variance.

```json
{
  "llm_threads": 0,
  "llm_auto_max_threads": 64,
  "llm_auto_window_size": 8,
  "llm_auto_warmup_items": 6,
  "llm_auto_increase_retry_ratio_max": 0.08,
  "llm_auto_increase_timeout_ratio_max": 0.0,
  "llm_auto_increase_latency_ratio_max": 1.35,
  "llm_auto_decrease_retry_ratio_min": 0.20,
  "llm_auto_decrease_timeout_ratio_min": 0.0,
  "llm_auto_decrease_latency_ratio_min": 1.9,
  "llm_auto_stable_throughput_factor": 0.95,
  "llm_auto_healthy_streak_required": 1,
  "llm_auto_cooldown_windows": 0
}
```

## Tuning Tips

- If threads keep dropping: lower `llm_auto_increase_latency_ratio_max` and/or lower `llm_auto_increase_retry_ratio_max`.
- If scaling is too slow on strong hardware: increase `llm_auto_max_threads` and lower `llm_auto_healthy_streak_required`.
- If oscillation occurs: increase `llm_auto_window_size` and `llm_auto_cooldown_windows`.
- If validation is unstable but AI Find is stable: keep global auto settings conservative and use fixed threads temporarily for problematic runs.

## Minimal Example

```json
{
  "llm_threads": 0,
  "llm_auto_max_threads": 24,
  "llm_auto_window_size": 10,
  "llm_auto_warmup_items": 8
}
```

Any omitted keys use built-in defaults.
