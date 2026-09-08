# Phase 2 dimension scores --- panel closed, 5/5 seats usable

Priorities from `contract.json`: D1, D2, D3, D6 mandatory; D4 high; D5 normal.

| Seat | D1 rigor | D2 domain | D3 coherence | D4 cross-disc | D5 writing | D6 fit |
|---|---|---|---|---|---|---|
| methodology | **block** (rep.) | — | **block** (rep.) | — | — | — |
| domain | — | **block** (rep.) | — | — | — | — |
| da | — | — | **block** (rep.) | — | — | — |
| perspective | — | — | — | **warn** | — | — |
| eic | — | — | — | — | **block** | **block** (rep.) |
| **worst per dimension** | **block** | **block** | **block** | **warn** | **block** | **block** |

`rep.` = `block_class: repairable`. No seat returned `fatal` on any dimension.

## Failure ladder
- **F1** (sev 95, reject) --- *any mandatory dimension has a fatal block*.
  Does **not** fire: every mandatory block is classed repairable.
- **F2** (sev 90, major_revision) --- *any mandatory dimension scores block*.
  **FIRES**, on all four mandatory dimensions independently (D1, D2, D3, D6).
- F3 (sev 70), F4 (sev 60), F5 (sev 40) also fire but are subsumed.
- F0 does not fire.

**editorial_decision = major_revision** (governed by F2, severity 90).

## Cross-seat convergence (independent seats, same defect)
1. **The transient/intermittent change literature is uncited** --- DA M2 and
   domain W1, arrived at from different directions (universal prediction vs.
   sequential analysis). Author-side web verification resolves every primary
   source both named. Three-way triangulated.
2. **Head-to-head episode numbers carry no uncertainty** --- DA C4 and
   methodology W13, both recomputing from `exp8_episodes.json`.
3. **72 grid points vs a shipped default of 24** --- perspective W2,
   eic W4, methodology W15.
4. **The DKW comparison figure is wrong** --- domain W6 and methodology W9,
   with rival diagnoses; methodology's (computed at n=1500) is exact.

## Conflicts adjudicated by the chair
- *0.051*: methodology's diagnosis is arithmetically exact and is adopted;
  the domain seat's is close but not exact. Both agree on the corrected 11.9x.
- *DA C1 vs methodology W1*: C1 argues the thesis fails because the prior-art
  changepoint mixture also achieves PFA 0.000. W1 argues no PFA verdict in the
  ablation is resolvable at 200 replicates (Wilson upper bound 0.0188). These
  do not conflict --- W1 strengthens C1. If neither arm is distinguishable from
  nominal at 200 reps, then the claim that only the new algorithm holds the
  level is *a fortiori* unsupported. Neither seat drew the joint conclusion:
  **the validity claim must be re-run for the changepoint mixture inside
  Experiment 1's 1500-replicate design**, which is the only properly powered
  validity experiment in the paper and currently runs the episodic arm alone.
