# A69 report correction

Date: 2026-09-21

The experiment owner confirmed that A69 was executed correctly. The earlier
audit mistook an error in the written resolution/component-source description
for an error in the experiment. All statements that describe A69 as a protocol
mismatch, invalid evidence, or something repaired by A74 are withdrawn.

No A69 result was recalculated or edited. The authoritative files are archived
under `results/a69_primary/`.

| Comparison | Delta AU-PRO@0.05 | Paired category bootstrap 95% CI | Wins |
|---|---:|---|---:|
| q99 mean - Global | +0.032664 | [0.007174, 0.063030] | 6/8 |
| q99 mean - Tiled | +0.022666 | [0.002874, 0.042679] | 5/8 |
| Tiny q99 mean - Global | +0.039571 | [0.010247, 0.072058] | 6/8 |

A69 q99 mean reaches 0.602419 overall and 0.582886 on tiny defects. It is the
primary AD2 result. A74 is retained as a separate uniform global-448/tiled-672
ablation and must not be described as repairing or replacing A69.
