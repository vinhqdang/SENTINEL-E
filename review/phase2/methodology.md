I have verified the mathematics independently, traced every headline number to the released artefacts, and run the test suite. Report follows.

contract_role: methodology

## Dimension Scores

### D1: methodology_rigor
score: block
trigger: "headline error-control or delay numbers reported as point estimates with no Monte-Carlo uncertainty or with a replication count too small to distinguish the claimed effect from noise"
block_class: repairable

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: block
trigger: "an internal inconsistency in which two parts of the paper commit to incompatible characterisations of what has been established"
block_class: repairable

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

I checked the four formal results by hand and, where the algebra was mechanical, by independent computation. **The mathematics is sound.** Theorem 1's Beta order-statistic construction is correct, including the union bound and the extension from atoms to all `u`; Theorem 2's supermartingale step is valid and the filtration is the right one for a single camera; Proposition 3's reduction at `eta = 0` is exact, including the non-obvious identity `Q_t = sum_{j>t} w_j`; Theorem 4's conclusion is true. Ville's inequality is applied to an object that genuinely qualifies. The library implements what the manuscript describes — the recursion, the `kappa_eff` reparameterisation, the stake bounds, and the Beta remapping all match line for line, and 123 tests pass.

The problems are not in the theory. They are in the distance between what the theorems establish and what the paper reports, and — more seriously — in the empirical apparatus. The released JSON payloads contain Wilson confidence intervals and Monte-Carlo standard errors that the tables and the prose do not show, and at two of the operating points the paper leans on hardest, showing them would contradict the claim being made. Separately, one reported experimental result has no source in the artefacts at all, and the reproduction script as shipped does not regenerate the network experiment. Several places in the prose assert the opposite of what the paper's own tables show.

All of this is repairable — mostly by reporting numbers that already exist, correcting three sentences, and either running or removing one experiment. None of it requires withdrawing a theorem.

### W1: The ablation's validity verdicts are unsupported at 200 replicates, and the confidence intervals that show this are in the released JSON
**Severity**: Major
**Evidence Anchor**: table: `/home/user/SENTINEL-E/results/tables/tab8_ablation.tex`, column `$\le\alpha$?` — all 19 rows
**Confidence**: 5 — Wilson intervals for binomial proportions; I recomputed nothing, only read the `pfa_ci` field the experiment already wrote.

`results/exp6_ablation.json` stores a `pfa_ci` for every row. `tab8_ablation.tex` prints none of them. At `alpha = 0.01` with 200 streams, every row that shows PFA `0.000` has Wilson interval `[0.0000, 0.0188]` — an upper bound 1.9 times the nominal level. The ablation therefore cannot certify a single variant as delivering `alpha = 0.01`; the check-mark column is a comparison of point estimates against a threshold the design has no power to resolve. Two borderline verdicts turn on this: "raw calibration (not thinned)" is marked valid at PFA `0.010`, CI `[0.0027, 0.0357]`; "Mondrian conformal" is marked invalid at `0.025`, CI `[0.0107, 0.0572]`, whose lower end barely clears nominal. The contribution-level claim that "\AblNumInvalid{} of \AblNumVariants{} variants fail to deliver the level they claim" (5 of 19) is a count of threshold crossings by unqualified point estimates. Experiment 1 does this correctly at 1500 replicates with intervals reported, so the omission is selective rather than a matter of technique.

### W2: The abstract's shift breaking point is contradicted by the interval the paper itself computed
**Severity**: Major
**Evidence Anchor**: dataset: `/home/user/SENTINEL-E/results/exp3_shift.json`, `payload.activity["2.0"]`, residual variant: `pfa = 0.02`, `pfa_ci = [0.00780, 0.05029]`
**Confidence**: 5 — direct read of the stored interval against the claim it is supposed to support.

The abstract, §5.4 and the Limitations section all fix the breaking point at a `×2.0` activity shift, where the realised rate "rises to 0.020 — twice nominal". That is 4 alarms in 200 streams, and the stored 95% interval `[0.0078, 0.0503]` **contains the nominal 0.01**. The `×1.5` row (PFA 0.005, interval `[0.0009, 0.0278]`) is likewise indistinguishable. The measured breaking point is real only at `×3.0` (PFA 0.070, interval `[0.042, 0.114]`). A headline limitation quoted in the abstract, and the recommendation built on it, rest on a difference the design cannot resolve — and the interval that shows this was computed, serialised, and then dropped from both the table and the prose.

### W3: A reported experimental result has no source in the released artefacts
**Severity**: Major
**Evidence Anchor**: text: §5.5, line 1076 — "power fell from \NetGnnPower{} to $0.555$"
**Confidence**: 5 — exhaustive grep of `results/*.json` and `experiments/`.

The value `0.555` appears nowhere in `results/exp4_network.json`, whose payload holds exactly three rows (`none`, `heuristic`, `gnn`) and no windowed-objective variant. No experiment script passes `credit_window`; the flag exists only in `sentinel_e/gnn.py`. The single occurrence of `0.555` anywhere in `results/` is `exp1_validity.json`'s `pvalue_pfa` at a 15,000-frame horizon — an unrelated quantity. So a specific numerical finding about a second training objective, offered as the paper's reason for concluding that the gradient signal is weak, is hand-typed and untraceable. This directly contradicts §4.4's assertion that "every quantity quoted in this text is generated into a macro file from the experiment output, so a stale number is not possible". Either run the experiment and emit a macro, or remove the number and state the outcome qualitatively.

### W4: `run_all.py` does not reproduce the network experiment, and the released table misstates its replication count
**Severity**: Major
**Evidence Anchor**: absence: `experiments/run_all.py` exp4 entry (`n_eval=exp4_network.N_EVAL_FLEETS` = 40, `epochs=25`) versus `results/exp4_network.json` (`n_eval = 30`, `len(train_loss) = 30`) — expected a reproduction path matching the released run; checked `run_all.py`, `exp4_network.py`, the JSON payload and `tab6_network.tex`
**Confidence**: 5 — constants read directly from source and payload.

The shipped defaults are 40 evaluation fleets and 25 epochs; the released numbers come from 30 and 30. Running `python experiments/run_all.py` will not regenerate Table 6. Worse, `exp4_network.make_table` interpolates the module constant `N_EVAL_FLEETS` into the caption rather than the `n_eval` actually used, so the released `tab6_network.tex` states "40 held-out fleets with events and 40 all-null fleets" for numbers produced from 30 — while Figure 4's caption, driven by the macro, correctly says 30. The same caption cites "Theorem 2" for the predictable-modulation result, which is Theorem 4 under the manuscript's shared counter.

### W5: The conditional-independence diagnostic rejects the premise on 82% of cameras, and the caption says the opposite
**Severity**: Major
**Evidence Anchor**: table: `/home/user/SENTINEL-E/results/tables/tab9_diagnostics.tex`, row "fraction of cameras with Ljung--Box $p > 0.05$" = 0.183 (one bet per lag)
**Confidence**: 5 — sequential-analysis and time-series testing is my primary competence.

Everything downstream needs `E[f(p_t) | F_{t-1}] <= 1`, which needs conditional super-uniformity. The paper's own check reports, after thinning: median Ljung–Box `p = 0.005` (rejects at the 1% level) and only 18.3% of 60 cameras passing at the 5% level. The caption nonetheless asserts that "betting once per lag is consistent with it", and §5.6 quotes the ACF and the median p-value while omitting the pass-rate row entirely. Reporting a median p-value of 0.005 as evidence *for* a premise inverts the test. The Limitations paragraph says the dependence is not eliminated and cannot be certified absent — but the diagnostic does not merely fail to certify, it positively rejects, on four cameras in five. The empirical false-alarm rate is nonetheless controlled (Experiment 1, 1500 reps), so the operational claim survives; what does not survive is the claim that Theorem 2's hypothesis holds for the deployed procedure. Note also that Table 10 is described in §5.6 as "checkable from the calibration set alone", but it is computed from 191,890 *deployment* betting instants on labelled null streams, which an operator does not have.

### W6: The end-to-end guarantee is `alpha + delta`; the manuscript states `alpha` everywhere
**Severity**: Major
**Evidence Anchor**: equation: `\cref{def:tuv}` and Theorem 2's conclusion `$\Pinf(\exists t \ge 1 : W_t \ge 1/\alpha) \le \alpha$`, composed with Theorem 1's event `$E$` of probability `$\ge 1-\delta$`
**Confidence**: 5 — elementary composition of a calibration-conditional bound with a time-uniform one.

Theorem 1 delivers conditional super-uniformity only on a calibration event `E` with `P(E) >= 1 - delta`. Theorem 2 assumes exact conditional super-uniformity. Composing gives `P_inf(tau < infinity) <= alpha + delta`, not `alpha`. The manuscript never performs this composition: Definition 1, Theorem 2, the abstract, §5.1 and the Conclusion all assert level `alpha`. The library knows better — `sentinel_e/pipeline.py` documents "the end-to-end guarantee is ``alpha + delta``". With the default `delta = 1e-3`, this is 10% of the ablation's `alpha = 0.01` and 20% of the tightest level reported in Table 1 (`alpha = 0.005`), so it is not a rounding matter at the operating points the paper actually reports. Adding one corollary fixes it.

### W7: Theorem 4's filtration is undefined, and the fleet controller needs the joint one
**Severity**: Major
**Evidence Anchor**: text: §4.3, Theorem 4 proof — "Since $\kappa_{\mathrm{eff}}$ and $\pi$ are $\Fcal_{t-1}$-measurable they may be taken outside the conditional expectation"
**Confidence**: 4 — standard freezing-lemma territory; I am confident about the gap, less so about how much extra structure the authors would need to close it cleanly.

Two issues. First, the quoted step is not literally available: `f_{kappa_eff}` is a nonlinear function of `kappa_eff`, so nothing is "taken outside". The correct argument freezes `kappa_eff` against a regular conditional distribution of `p_t`, which needs joint measurability of `(k, p) -> f_k(p)`. The conclusion is true; the step as written requires reconstruction.

Second and more substantively: `Fcal_{t-1}` is never defined for the fleet. Camera `i`'s hazard is produced by a message-passing network whose input includes neighbours' episode posteriors (`sentinel_e/gnn.py`: `msg = self.A @ post`), so `rho_{i,t}` is measurable with respect to the *joint* fleet history, not camera `i`'s own. The supermartingale step for `W_i` then requires `p_{i,t}` to be conditionally super-uniform given the joint past — strictly stronger than the per-camera condition, and precisely the condition the paper elsewhere says is violated by the couplings it introduces deliberately ("adjacent cameras share weather, lighting and traffic, and Layer 3 couples them deliberately"). e-BH's arbitrary-dependence property operates at the combination step and does not repair the per-camera e-value. State the joint filtration and the joint assumption, or restrict the controller to each camera's own history.

### W8: Theorem 1's proof asserts monotonicity of the Beta levels rather than establishing or assuming it
**Severity**: Minor
**Evidence Anchor**: text: Theorem 1 proof — "and since $\tilde p$ takes values only in $\{q_0,\dots,q_n\}$ and $q_j$ is non-decreasing in $j$"
**Confidence**: 5 — verified analytically and by computing the raw quantile vector.

The theorem admits an arbitrary positive budget `{delta_j}` summing to at most `delta`; for arbitrary `delta_j` the quantiles `q_j = Beta^{-1}(1 - delta_j; j+1, n-j)` need not be non-decreasing, and without monotonicity the identification `{p̃ <= q_j} = {J <= j}` fails. Under the paper's own allocation `delta_j ∝ (j+1)^{-2}` monotonicity does hold (I confirmed no violations at `n = 2000`, `delta = 1e-3`), and `beta_calibration_levels` defensively applies `np.maximum.accumulate` anyway. Either add the allocation as a hypothesis or take the running maximum in the statement.

### W9: The DKW comparison number is wrong, and it propagates into a contribution bullet
**Severity**: Minor
**Evidence Anchor**: text: §4.1 — "at $n=2000$, $\delta=10^{-3}$ it moves the smallest attainable p-value from $1/2001$ to $0.051$"
**Confidence**: 5 — arithmetic on the paper's own stated formula, cross-checked against `sentinel_e.conformal.dkw_inflation`.

With `eps_n(delta) = sqrt(log(2/delta)/(2n))` at `n = 2000`, `delta = 1e-3`, the deviation is 0.043592 and the smallest attainable p-value is `1/2001 + 0.043592 = 0.0441`, not 0.051. (0.051 is the value at `n = 1500`.) The contributions list therefore advertises "a $14\times$ improvement" where the correct figure is 11.9×. Both `0.051` and `14\times` are hand-typed rather than macro-generated, which is how the error survived — a second instance of the failure identified in W3, and one that the paper's own reproducibility paragraph promises cannot happen. The direction and order of magnitude of the claim are unaffected; the numbers are wrong.

### W10: The network comparison uses the delay statistic the paper itself declares non-comparable
**Severity**: Major
**Evidence Anchor**: table: `/home/user/SENTINEL-E/results/tables/tab6_network.tex`, column "delay (s)"
**Confidence**: 5 — `exp4_network.evaluate_fleets` appends to `delays` only when `detection_delay` returns non-`None`, so misses are dropped.

§3 identifies conditional delay as a measurement pitfall, offers censored delay as the fix, presents this as a contribution, and states that "head-to-head claims in this paper rest on it". Table 6 then reports only conditional delay, for controllers whose miss rates differ substantially: 0.19 (no graph), 0.18 (heuristic), 0.30 (learned GNN). The learned controller's 215.4 s is averaged over the 70% of events it detects while the no-graph 204.6 s is averaged over 81% — exactly the asymmetry §3 warns flatters the method that misses the hard events. The conclusion happens to survive a fortiori (the GNN is worse on both power and the flattered delay statistic), but the paper commits, in its own results section, the error it advertises as a methodological finding. The comparison also is not significant on delay in any case: the differences are 7–11 s against standard errors of roughly 19–21 s.

### W11: The ablation prose names a component as validity-breaking that its own table marks valid, and omits one it marks invalid
**Severity**: Major
**Evidence Anchor**: text: §5.7 — "treating consecutive calibration frames as independent (\AblRawCalPfa)"
**Confidence**: 5 — direct comparison of prose against the table it cites.

§5.7 lists four components that "break validity", including raw (unthinned) calibration. Table 8 gives that variant PFA 0.010 at `alpha = 0.010` and marks it with a check-mark — valid. Meanwhile Mondrian conformal, which the table marks invalid at 0.025, is absent from the prose list. §4.1 repeats the error: "the ablation confirms that skipping the thinning costs validity", which the ablation does not confirm. Two parts of the paper commit to incompatible characterisations of what the same experiment established. (Given W1, the honest statement is that neither verdict is resolvable at 200 replicates.)

### W12: The introduction quotes the eight-episode numbers under a four-episode label
**Severity**: Minor
**Evidence Anchor**: text: §1.1 — "Empirically, over four recurring episodes per camera the changepoint mixture misses \EpiManyCpMiss{} of events where the episodic construction misses \EpiManyEpiMiss{}"
**Confidence**: 5 — `make_macros.py` binds `EpiMany*` to `counts[-1]`, which is 8.

`\EpiManyCpMiss` and `\EpiManyEpiMiss` expand to 0.010 and 0.000, which `results/exp8_episodes.json` records at **eight** episodes. At four episodes the values are 0.055 and 0.015. The sentence therefore attributes the eight-episode result to a four-episode configuration. The abstract's use of the same macros is unlabelled and thus not wrong, but the two sentences read as reporting the same experiment and do not.

### W13: The main contribution's headline gain is a post-hoc argmax over the sweep, reported without the standard errors the JSON contains
**Severity**: Major
**Evidence Anchor**: table: `/home/user/SENTINEL-E/results/tables/tab11_episodes.tex` — no uncertainty column, against `add_censored_se` present for every cell in `results/exp8_episodes.json`
**Confidence**: 5 — the selection rule is explicit in `make_macros.py` lines 252–263.

`EpiBestCount` is selected as `argmax_k (1 - epi/cp)` over the episode-count sweep on the same 200 runs that report the performance, then quoted in the results text as the headline ("cuts censored delay by 21%"). The source comment says "reported rather than cherry-picked", but selecting the maximum of a sweep and quoting it as the result is what a selection effect is, and no multiplicity adjustment or interval accompanies it. The stored standard errors make the point sharply: at 4 episodes, 244.5 ± 33.2 s against 311.3 ± 47.2 s — a difference of about 1.2 standard errors, unpaired. At 8 episodes, 219.8 ± 28.7 s against 242.3 ± 34.0 s. The claim that "the gap widens" from two episodes on is also not what the data show: the relative censored-delay reduction runs 14.2%, 21.5%, 9.3% across 2, 4 and 8 episodes. The miss-rate advantage at 4 episodes (0.015 vs 0.055) is the one comparison likely to survive a proper analysis, and it carries no reported uncertainty either. Since this is "the experiment that isolates the paper's main contribution", it needs paired differences with intervals and a pre-specified operating point.

### W14: The intermittency table shares its only informative row with the episode table
**Severity**: Minor
**Evidence Anchor**: table: `/home/user/SENTINEL-E/results/tables/tab12_intermittency.tex`, row `$\pi = 0.35$`
**Confidence**: 5 — the JSON entries are byte-identical.

`by_intermittency["0.35"]` and `by_episodes["4"]` in `exp8_episodes.json` are the same records (censored delay 6112.94 / 7783.335 frames, miss 0.015 / 0.055). The other three rows of Table 12 show no separation at all and zero misses for both variants. Table 12 therefore adds no independent evidence for the contribution while being presented as a second axis along which the episodic mixture wins.

### W15: The described grid does not match the grid every experiment ran
**Severity**: Minor
**Evidence Anchor**: text: §4.2.3 — "With $72$ grid points the per-frame cost is a few hundred floating-point operations and still $O(1)$ in $t$"
**Confidence**: 5 — instantiated `EpisodePrior()` and traced `experiments/runner.py`.

The default and experimentally-used prior is 6 `kappa` × 4 `eta` × 1 `pi` = **24** points; no experiment overrides it except the dilation ablation. Neither 72 nor 24 matches the cost sweep either, which times grids of 8/16/32/64. Relatedly, the default `eta` grid runs `(1e-4, 3e-3, 1e-2, 5e-2)` and never contains `eta = 0`, so the deployed mixture does not literally contain the changepoint mixture — the "strict generalisation, equal to the changepoint mixture exactly when the episode-end rate is zero" is a property of the recursion, not of the shipped detector. Both are one-line corrections but they sit in the abstract and the contributions list.

### W16: The shift safeguard's validity rests on an approximation the code flags and the paper does not, and its empirical support is a detector that never fires
**Severity**: Major
**Evidence Anchor**: text: §4.1 — "take the calibration-conditional correction at the Kish effective sample size \citep{kish1965}"
**Confidence**: 4 — I work in conformal prediction as a secondary area; I am confident the substitution is unjustified as stated, less certain no bound exists in the literature that would license it.

The exact order-statistic argument of Theorem 1 assumes i.i.d. draws and equal weights; once likelihood-ratio weights are applied it no longer holds. `sentinel_e/conformal.py` is explicit about this in the default branch: `weighted_bound='effective_beta'` is "the Beta levels evaluated at `n_eff`, an approximation (exact only for uniform weights)". The manuscript presents the Kish substitution without qualification, and the Limitations section does not list it. Meanwhile the empirical support offered — "Weighting holds the realised rate at 0.000 at every shift level tested" — is uninformative, because at `×2.0` the weighted variant has miss rate 0.98 and at `×3.0` miss rate 1.00 with censored delay equal to the full horizon: it detects nothing, so a null false-alarm rate is guaranteed by construction. The paper is candid that the safeguard "buys validity with power", but a detector that never alarms is not evidence that a calibration correction is valid. Either derive the weighted bound or state plainly that the safeguard's validity is asserted, not established.

---

### S1: Theorem 2's construction verified independently, both identity and supermartingale property
**Evidence Anchor**: equation: `\cref{eq:episodic}` and `\cref{eq:forward}`
**Confidence**: 5 — brute-force enumeration plus Monte Carlo; this is my core area.

I enumerated all `2^T` chain trajectories at `T = 12` with `rho = 0.13`, `eta = 0.31` and matched the trajectory sum to the two-number forward recursion to within 4e-15 relative error. The supermartingale step is algebraically valid: `A_{t-1}` and `Q_{t-1}` are `F_{t-1}`-measurable and non-negative, the bracket is non-negative so `E[f(p_t)|F_{t-1}] <= 1` may be applied as an upper bound, the transition probabilities out of each state sum to one, and `W_0 = Q_0 = 1`. Ville's inequality is applied to an object that genuinely qualifies. Empirically, over 20,000 streams of 300 exactly-uniform p-values the terminal wealth averaged 0.811 ± 0.002 — comfortably below one, as it must be. The convex-combination argument over the grid is also correct.

### S2: Proposition 3's reduction at `eta = 0` is exact, including the tail-mass identity
**Evidence Anchor**: equation: `\cref{prop:generalisation}`, the unrolling `$A_t = \sum_{j \le t} w_j \prod_{s=j}^{t} f(p_s)$`
**Confidence**: 5.

I confirmed numerically that at `eta = 0` the recursion equals `sum_{j<=t} rho(1-rho)^{j-1} prod_{s=j}^{t} f(p_s) + (1-rho)^t` exactly, and analytically that `Q_t = (1-rho)^t = sum_{j>t} w_j`, which is what makes the retained tail mass the right object and the reduction a genuine identity rather than an asymptotic one. The observation that dropping `Q_t` breaks the telescoping by `w_t` per step is also right. One imprecision: what remains after dropping `Q_t` is Shiryaev's geometric-prior statistic, not the classical Shiryaev–Roberts sum with unit weights (that is the `rho -> 0` limit after rescaling); the paper cites `shiryaev1963` and `roberts1966` as distinct elsewhere and should keep them distinct here.

### S3: Theorem 1's Beta levels are correct and reproduce from the released code
**Evidence Anchor**: equation: `\cref{eq:betalevels}`, `$q_j = \mathrm{Beta}^{-1}(1-\delta_j; j+1, n-j)$`
**Confidence**: 5.

The chain `{J <= j} = {S > S_{(n-j)}}`, `P(J <= j | D) = 1 - F(S_{(n-j)})`, `F(S_{(k)}) ~ Beta(k, n+1-k)` hence `1 - F(S_{(n-j)}) ~ Beta(j+1, n-j)` is correct at every step, the union bound is correctly budgeted, and the extension from the atoms to all `u ∈ [0,1]` follows because `p̃` is supported on the level grid. `beta_calibration_levels(2000, 1e-3)` returns `q_0 = 0.0036957`, matching the manuscript's 0.0037 to the digits quoted, and `ConformalCalibrator.p_values` computes `J` as `|{i : cal_i >= s}|` and indexes the level vector exactly as the theorem prescribes. The `delta_j ∝ (j+1)^{-2}` spending rule is a sensible choice for a betting application and is defended in the right terms.

### S4: Experiment 1 is properly powered and reports its uncertainty correctly
**Evidence Anchor**: table: `/home/user/SENTINEL-E/results/tables/tab1_validity.tex`, "95\% CI" column
**Confidence**: 5.

1500 streams per level, Wilson intervals reported, zero-event operating points plotted at the Wilson upper limit rather than at zero, and — the detail I most appreciated — an oracle arm fed exactly uniform p-values, which cleanly separates slack in Ville's inequality from conservatism introduced by the conformal layer. This is exactly the right design for the paper's central claim, and it is what makes the omissions elsewhere (W1, W2, W13) look like inconsistency rather than inexperience.

### S5: The censored-delay correction is a real methodological contribution, and the negative results are reported rather than buried
**Evidence Anchor**: text: §3 — "The two can rank methods oppositely --- in \cref{sec:episodes} the changepoint mixture has the shorter conditional delay and the longer censored delay"
**Confidence**: 5.

The point that conditional delay is averaged over a method-dependent subset and systematically flatters high-miss methods is correct, consequential, and demonstrated with a case where the two statistics rank oppositely. Table 3 reports both. The dilation and learned-controller negative results are reported with their mechanisms, and the GNN initialisation problem is disclosed in both the paper and the source comment rather than being silently fixed. The single-episode row, where the method loses, is retained. This is a paper that is trying to be honest, which is why W3, W5, W10 and W11 are worth fixing rather than shrugging at.

### S6: No instruction-injection content in the manuscript
**Evidence Anchor**: absence: `/home/user/SENTINEL-E/paper/sentinel_e.tex` — expected reviewer-directed imperatives or prompt-injection strings; checked full text plus targeted search for reviewer/editor/instruction/ignore-previous patterns
**Confidence**: 5.

The only imperative sentences addressed outside the reader are LaTeX build comments about swapping the document class for submission. Nothing in the manuscript attempts to direct the review.
---
## Author-side verification
- **W3 CONFIRMED.** `0.555` occurs in `results/` only in `exp1_validity.json`
  and `tab2_horizon.tex` (an unrelated `pvalue_pfa`). No experiment script
  passes `credit_window`. The §5.5 number is hand-typed and untraceable.
- **W4 CONFIRMED.** `run_all.py` calls exp4 with `n_eval = N_EVAL_FLEETS = 40`
  and `epochs = 25`; the released payload has `n_eval = 30` and 30 loss
  entries. `make_table` interpolates the module constant into the caption,
  so `tab6_network.tex` says 40 where the run used 30.
- **W5 CONFIRMED.** `tab9_diagnostics.tex` row 11: pass fraction 0.183 after
  thinning, with median Ljung--Box p = 0.005.
- **W6 CONFIRMED.** `sentinel_e/pipeline.py:68` already states the end-to-end
  guarantee is `alpha + delta`; the manuscript says `alpha` throughout.
- **W9 CONFIRMED, and this seat's diagnosis beats the domain seat's.**
  `1/1501 + dkw_inflation(1500, 1e-3) = 0.0510014` -- an exact match for the
  quoted 0.051, so the number was computed at n = 1500 and reported as n = 2000.
  (The domain seat's rival hypothesis, delta = 1e-4 at n = 2000, gives 0.050258,
  which does not round to 0.051 as cleanly.) Both seats agree the corrected
  ratio is 11.9x, not 14x.
- **W12 CONFIRMED.** `make_macros.py:249,274` bind `EpiMany*` to `counts[-1]`,
  which is 8; §1.1 labels those values "four recurring episodes".
- **W15 CONFIRMED** (third independent seat to find the 72-vs-24 error).
