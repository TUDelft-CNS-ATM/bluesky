# Aircraft performance models

Aircraft performance — thrust, drag, fuel flow, climb/descent rates, speed
envelopes — is handled by a **replaceable** performance model
(`bs.traf.perf`), selected by the `performance_model`
[setting](../user-guide/configuration.md) (default `'openap'`) or switched
at runtime with the `IMPL`/`IMPLEMENTATION` command.

Three models ship with BlueSky, all under `bluesky/traffic/performance/`:

## OpenAP (default)

[OpenAP](https://github.com/junzis/openap) is an open-source, open-data
aircraft performance model, and is a core BlueSky dependency, so it works
out of the box with no extra setup.

## BADA

EUROCONTROL's BADA (Base of Aircraft Data) 3.x model. **BADA data is
license-restricted and is not distributed with BlueSky** — if you have
access to a BADA license, place the data files under the `BADA`
subdirectory of your BlueSky performance resources, then select the model
with `IMPL PERFBASE BADA` (or the equivalent `performance_model` setting).

## Legacy (`legacy`)

BlueSky's own original, simpler performance model (`performance_model =
'legacy'`), kept for backwards compatibility with older scenarios and
studies.

## Switching models

```text
IMPL PERFBASE BADA
```

lists and selects among the registered implementations for the `PerfBase`
replaceable entity — the same mechanism used for [conflict
resolution](conflict-detection-resolution.md) and other replaceable
subsystems. Use `IMPL` with no arguments to list all replaceable entities
and their currently selected implementation.

## Per-aircraft engine/type selection

Aircraft performance is looked up per aircraft type (the `type` argument to
`CRE`); the `ENG` command lets you override the specific engine variant for
an aircraft where the model supports multiple engine options.
