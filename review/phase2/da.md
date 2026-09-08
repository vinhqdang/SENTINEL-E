# Phase 2 — Devil's Advocate seat

contract_role: da
Scoring-plan dissent: none.

## Dimension scores
- D1 methodology_rigor: not_assessed
- D2 domain_accuracy: not_assessed
- D3 argumentative_coherence: **block** (block_class: repairable)
  - trigger: "A load-bearing claim in the abstract, contributions list, or
    conclusion is not entailed by the evidence offered for it"
- D4 cross_disciplinary_relevance: not_assessed
- D5 writing_and_structure: not_assessed
- D6 venue_fit_and_contribution: not_assessed

## CRITICAL
| # | Issue | Anchor |
|---|---|---|
| C1 | Thesis negated by own ablation: the prior-art changepoint mixture on the identical conformal front end also holds the level (PFA 0.000), so closing the optional-stopping gap did not require the new algorithm | sec:conclusion "required a new algorithm rather than an application of existing ones" |
| C2 | Uniqueness claim rests on a baseline set from which the only competitor with the same formal guarantee was removed | abstract "the only method that holds its nominal false-alarm level at all" |
| C3 | The isolation experiment does not isolate: episodic arm 6 kappa x 4 eta = 24 pts; changepoint arm 32 log-spaced kappa. Ablation's own single-bet row shows a grid change alone moves delay 244.5->316.8 s, the size of the claimed episode effect | sec:episodes "same conformal layer, same betting grid" |
| C4 | No uncertainty on any episodic-vs-changepoint difference; reconstructed unpaired differences reach at most 1.16 SE; headline miss reduction is 0/200 vs 2/200, Fisher p=0.50 | tab11, tab12 |
| C5 | Advantage regime (4-8 episodes, ~4 min gaps, 60 min stream) contradicts the weeks-scale motivating story; in the 1-episode row the episodic mixture is worse on both quantities | abstract "recurs weeks later at the same site" |

## MAJOR
| # | Issue |
|---|---|
| M1 | Computed miss rate is stream-level, not the per-segment quantity defined in sec:problem; no per-episode metric exists, so the accumulate-across-episodes mechanism is unmeasured |
| M2 | Novelty framing asserts an unoccupied axis that is an established literature (transient change detection: Guepie/Fillatre/Nikiforov 2012; intermittent change of unknown duration: Sokolov/Spivak/Tartakovsky 2023; epidemic changepoint: Levin & Kline 1985, Fisch/Eckley/Fearnhead); construction is the fixed-share / expert-HMM mixture (Herbster & Warmuth 1998; Koolen & de Rooij; van Erven/Grunwald/de Rooij switch distribution). None cited |
| M3 | Contribution bullet 2 (episode posterior as inter-camera message) is justified solely by the graph layer that sec:network refutes; contributions list not revised |
| M4 | All headline experiments share one simulator configuration departing from library defaults in the three directions that favour the method; the same 200 seeded runs are reprinted across sec 6.2, 6.4, 6.7 |
| M5 | Episodic-vs-changepoint comparison run only at a threshold whose realised PFA is 0.000, while the paper advises deploying at an empirically calibrated threshold |
| M6 | Stated 72 grid points vs shipped default 24 |
| M7 | The transferable conclusion belongs to the conformal front end plus any exact e-process, not to the episode axis |

## Repair set proposed by the seat
Restate the thesis narrowly; put the changepoint mixture into Table 3 and
Figure 2; re-run sec 6.2 with one shared betting grid and paired differences
with intervals; add a per-episode detection metric or drop the mechanism claim;
sweep the episode comparison across realised FARs; add a genuine
persistent-change arm; cite and position against transient/intermittent/
epidemic changepoint and fixed-share; propagate the negatives into the
contributions list.

## Author-side verification of the Devil's Advocate findings
Every CRITICAL is adjudicated here (Iron Rule #4); MAJORs checked where
mechanically verifiable.

- **C1 CONFIRMED.** `results/exp6_ablation.json`: the `changepoint mixture
  ($\eta=0$)` row has `pfa = 0.0` against nominal 0.01, identical to
  SENTINEL-E (full). The prior art closes the optional-stopping gap on this
  front end. The conclusion's "required a new algorithm" is not entailed;
  only a *power* claim is.
- **C2 CONFIRMED.** `results/tables/tab3_delay_far.tex` lists seven methods
  (SENTINEL-E, fixed threshold, p-value threshold, CUSUM, Shiryaev--Roberts,
  parametric e-detector, E-SHIFT). The changepoint mixture is announced at
  sentinel_e.tex:796 as a baseline and appears in no row of Table 3 or
  Figure 2, where the abstract's uniqueness claim is established.
- **C3 CONFIRMED.** `EpisodePrior.kappa = (0.02, 0.05, 0.12, 0.28, 0.55, 0.85)`
  x `eta` (4) x `pi` (1) = 24 points, class `EpisodicEDetector`. The
  changepoint arm is `EDetector` + `ChangepointPrior` with `n_grid = 32`
  log-spaced kappa (pipeline.py:96, 137-146). The arms do **not** share a
  betting grid, contrary to sec:episodes. Magnitude check: collapsing the
  kappa grid alone ("single bet aggressiveness", still episodic) moves
  censored delay 244.5 s -> 316.8 s and miss 0.015 -> 0.045; the whole
  claimed episode-axis effect is 244.5 s -> 311.3 s, 0.015 -> 0.055.
  The two effects are the same size.
- **C4 CONFIRMED.** Recomputed unpaired differences (censored delay,
  changepoint minus episodic) from `results/exp8_episodes.json`:
  1 ep -55.7 +- 138.1 s (-0.40 SE); 2 ep +80.0 +- 98.0 s (+0.82 SE);
  4 ep +66.8 +- 57.7 s (+1.16 SE); 8 ep +22.6 +- 44.5 s (+0.51 SE).
  Nothing reaches 1.2 SE. Neither tab11 nor tab12 carries an interval.
  Note the arms share seeds, so a *paired* analysis is available to us and
  is the correct repair rather than a concession.
- **C5 CONFIRMED.** 1-episode row: episodic censored delay 1257.3 s vs
  changepoint 1201.6 s, miss 0.445 vs 0.425 -- episodic nominally worse on
  both (within noise). The caption of fig:episodes asserts the two "agree".
- **M1 CONFIRMED.** `metrics.detection_delay` scans `alarms[change_point:]`
  from the *first* onset and `miss_rate = misses / n_signal` with `n_signal`
  the stream count. With `n_episodes > 1` this is stream-level, not the
  per-segment definition given in sec:problem. No per-episode metric exists.
- **M2 CONFIRMED.** `paper/refs.bib` (45 entries) contains no work on
  transient, intermittent or epidemic change detection, and none on
  fixed-share / switching-expert mixtures. The cited alternatives named by
  the seat must themselves be verified before being added.
- **M3 CONFIRMED.** Contribution bullet 2 (sentinel_e.tex:218-223) grounds the
  episode posterior's value in the message-passing layer that sec:network
  then reports as worthless; the bullet is not revised.
- **M4 CONFIRMED.** `seed0 = 10_000` (runner.py:192) throughout, and
  `add = 5140.040609137056` recurs in exp2, exp3, exp6 and exp8. Headline
  config `n_episodes=4, event_length=800, episode_gap=6000,
  event_intermittency=0.35` (exp2:33, exp3:78/92, exp6:89) departs from the
  library defaults `n_episodes=1, event_length=1500, event_intermittency=0.5`
  in the three directions the exp8 sweep shows favour the episodic arm.
- **M6 CONFIRMED** (already logged from the perspective seat).

### Verification of the references the seat proposes for M2
Checked before any of them is added to `refs.bib`; a hallucinated citation
would be worse than the omission.

- **Real, and exactly as described.** Guepie, Fillatre & Nikiforov,
  "Sequential detection of transient changes", *Sequential Analysis* 31(4),
  528--547 (2012), doi 10.1080/07474946.2012.719443 — window-limited CUSUM for
  a change of finite *known* duration, with latent detection counted as a miss.
- **Real, and closer to our problem than the seat states.** Sokolov, Spivak &
  Tartakovsky, "Detecting an intermittent change of unknown duration",
  *Sequential Analysis* 42(3), 269--302 (2023), arXiv:2210.17342. This is our
  problem statement in its title. Its existence is decisive against the paper's
  "an axis the change-detection literature has left fixed at infinity".
- **Real.** Herbster & Warmuth, "Tracking the best expert", *Machine Learning*
  32(2), 151--178 (1998) — the fixed-share algorithm.
- **Partly mis-attributed by the seat.** "Epidemic changepoint detection in the
  presence of nuisance changes" (arXiv:2008.08240) is Juodakis & Marsland, not
  Fisch, Eckley & Fearnhead. The Fisch/Eckley/Fearnhead contribution in this
  area is CAPA, "A linear time method for the detection of collective and point
  anomalies". Both are relevant; the attribution must be corrected on the way in.
- **Additional hits the seat did not name, both on point.** Tartakovsky et al.,
  "Optimal sequential detection of signals with unknown appearance and
  disappearance points in time" (arXiv:2102.01310); and "Score-based change
  point detection via tracking the best of infinitely many experts"
  (arXiv:2408.14073), which is the fixed-share/change-detection bridge itself.

Net effect: M2 is not merely substantiated, it is understated. The related-work
repair is larger than the seat's list implies.
