# Ollama Settings Reference

Navigation: [Project README](../README.md) | [Docs Index](README.md) | [Usage Guide](usage.md)

This page explains ImageTagger Ollama settings with a focus on auto thread mode.

## Scope

These settings affect Generate, Validate, and AI Find when `ollama_threads` is set to `0` (auto mode).

If `ollama_threads` is greater than `0`, fixed thread mode is used and auto thresholds are ignored.

## Core Settings

- `ollama_server`: Ollama server URL.
- `ollama_model`: active model name.
- `ollama_max_resolution_mpx`: input image downscale target in megapixels.
- `ollama_threads`: thread count. Use `0` for auto mode.

## Auto Mode Thresholds

- `ollama_auto_max_threads`: hard upper bound for auto mode.
- `ollama_auto_window_size`: completions per control window.
- `ollama_auto_warmup_items`: minimum completions before scale-up.
- `ollama_auto_increase_retry_ratio_max`: maximum retry ratio allowed for scale-up.
- `ollama_auto_increase_timeout_ratio_max`: maximum timeout ratio allowed for scale-up.
- `ollama_auto_increase_latency_ratio_max`: maximum latency inflation allowed for scale-up.
- `ollama_auto_decrease_retry_ratio_min`: retry ratio threshold to scale down.
- `ollama_auto_decrease_timeout_ratio_min`: timeout ratio threshold for aggressive scale-down.
- `ollama_auto_decrease_latency_ratio_min`: latency inflation threshold to scale down.
- `ollama_auto_stable_throughput_factor`: required throughput floor vs previous window for scale-up.
- `ollama_auto_healthy_streak_required`: healthy windows required before increasing threads.
- `ollama_auto_cooldown_windows`: windows to wait after scale-down before increasing again.

## Presets

Use these as starting points. Keep `ollama_threads` at `0` for all presets.

### Safe Preset

Best for mixed workloads, unstable servers, or conservative operation.

```json
{
  "ollama_threads": 0,
  "ollama_auto_max_threads": 16,
  "ollama_auto_window_size": 10,
  "ollama_auto_warmup_items": 10,
  "ollama_auto_increase_retry_ratio_max": 0.03,
  "ollama_auto_increase_timeout_ratio_max": 0.0,
  "ollama_auto_increase_latency_ratio_max": 1.15,
  "ollama_auto_decrease_retry_ratio_min": 0.10,
  "ollama_auto_decrease_timeout_ratio_min": 0.0,
  "ollama_auto_decrease_latency_ratio_min": 1.40,
  "ollama_auto_stable_throughput_factor": 1.00,
  "ollama_auto_healthy_streak_required": 2,
  "ollama_auto_cooldown_windows": 1
}
```

### Balanced Preset

Current default behavior in the app.

```json
{
  "ollama_threads": 0,
  "ollama_auto_max_threads": 48,
  "ollama_auto_window_size": 6,
  "ollama_auto_warmup_items": 4,
  "ollama_auto_increase_retry_ratio_max": 0.18,
  "ollama_auto_increase_timeout_ratio_max": 0.0,
  "ollama_auto_increase_latency_ratio_max": 1.35,
  "ollama_auto_decrease_retry_ratio_min": 0.30,
  "ollama_auto_decrease_timeout_ratio_min": 0.0,
  "ollama_auto_decrease_latency_ratio_min": 2.0,
  "ollama_auto_stable_throughput_factor": 0.90,
  "ollama_auto_healthy_streak_required": 1,
  "ollama_auto_cooldown_windows": 0
}
```

### Aggressive Preset

For fast GPUs and models where you want faster ramp-up and accept more variance.

```json
{
  "ollama_threads": 0,
  "ollama_auto_max_threads": 64,
  "ollama_auto_window_size": 8,
  "ollama_auto_warmup_items": 6,
  "ollama_auto_increase_retry_ratio_max": 0.08,
  "ollama_auto_increase_timeout_ratio_max": 0.0,
  "ollama_auto_increase_latency_ratio_max": 1.35,
  "ollama_auto_decrease_retry_ratio_min": 0.20,
  "ollama_auto_decrease_timeout_ratio_min": 0.0,
  "ollama_auto_decrease_latency_ratio_min": 1.9,
  "ollama_auto_stable_throughput_factor": 0.95,
  "ollama_auto_healthy_streak_required": 1,
  "ollama_auto_cooldown_windows": 0
}
```

## Tuning Tips

- If threads keep dropping: lower `ollama_auto_increase_latency_ratio_max` and/or lower `ollama_auto_increase_retry_ratio_max`.
- If scaling is too slow on strong hardware: increase `ollama_auto_max_threads` and lower `ollama_auto_healthy_streak_required`.
- If oscillation occurs: increase `ollama_auto_window_size` and `ollama_auto_cooldown_windows`.
- If validation is unstable but AI Find is stable: keep global auto settings conservative and use fixed threads temporarily for problematic runs.

## Minimal Example

```json
{
  "ollama_threads": 0,
  "ollama_auto_max_threads": 24,
  "ollama_auto_window_size": 10,
  "ollama_auto_warmup_items": 8
}
```

Any omitted keys use built-in defaults.
