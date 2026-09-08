# Editorial Decision Letter

**Manuscript.** SENTINEL-E: Anytime-Valid Sequential Detection for Streaming
Illegal Dumping Surveillance
**Venue.** *Neural Computing and Applications*, topical collection "Emerging
Trends in Smart Waste Monitoring and Environmental Surveillance"
**Panel.** 5 seats (methodology, domain, devil's advocate, perspective,
journal-fit/EIC), paper-blind Phase 1 pre-commitment, paper-visible Phase 2.
**Contract.** `reviewer/reviewer_full/v2`, baseline v3.20.0.

## Decision: MAJOR REVISION

Determined mechanically. Failure condition **F2** (severity 90, *any mandatory
dimension scores block*) fires on all four mandatory dimensions independently:
D1 methodology rigor (methodology seat), D2 domain accuracy (domain seat),
D3 argumentative coherence (devil's advocate **and** methodology), D6 venue fit
and contribution (EIC). F1 (reject) does not fire: no seat returned a fatal
block, and every seat that blocked classed its block **repairable**.

## What the panel agrees is right

The formal core survives adversarial checking. The methodology seat verified
Theorem 1's Beta order-statistic construction step by step, brute-forced all
2^12 chain trajectories against the two-number forward recursion to 4e-15
relative error, confirmed the supermartingale property empirically (terminal
wealth 0.811 +/- 0.002 over 20,000 uniform streams), and confirmed
Proposition 3's reduction at eta = 0 including the tail-mass identity
Q_t = sum_{j>t} w_j. **No theorem is withdrawn.** The library implements what
the manuscript describes, and 123 tests pass.

Two domain claims that looked most likely to fail survived a hostile search.
The domain seat read all seven published WACV 2026 WasteVision IWDD papers plus
the organisers' contest overview and confirmed that none reports a false-alarm
rate over a monitoring horizon; and it found no application of e-process or
conformal-martingale machinery to waste, dumping or pollution surveillance.
The Mivia-IWDD-500 description verifies in every checkable particular.

Experiment 1 is properly powered (1500 replicates, Wilson intervals, an oracle
arm separating Ville slack from conformal conservatism). The censored-delay
correction is a real methodological contribution. The negative results are
reported rather than buried. Multiple seats said so unprompted.

## Why it blocks anyway

**The paper's reporting does not meet the standard its own Experiment 1 sets.**
The released JSON already contains Wilson intervals and Monte-Carlo standard
errors that the tables and prose omit — and at the two operating points the
paper leans on hardest, showing them contradicts the claim. The abstract's shift
breaking point (x2.0, PFA 0.020) has stored interval [0.0078, 0.0503], which
**contains nominal**. The ablation's 19 validity verdicts rest on point
estimates whose Wilson upper bound is 0.0188 against alpha = 0.01, so the design
cannot certify any of them. The main contribution's headline gain is an argmax
over a sweep, quoted without interval, on differences that reach at most
1.2 unpaired standard errors.

**The novelty framing rests on a literature claim that is false.** Two seats,
reasoning from different fields, independently found that "an axis the
change-detection literature has left fixed at infinity" is occupied: transient
change detection (Guepie/Fillatre/Nikiforov 2012, 2017), intermittent change of
unknown duration (Sokolov/Spivak/Tartakovsky 2023 — our problem in its title),
two-state HMM quickest detection (Fuh & Mei 2015), and, on the construction
side, fixed-share and switching-expert mixtures (Herbster & Warmuth 1998). The
chair verified every primary source. The defensible claim — new *within the
anytime-valid e-process framework* — is intact and is the one to make.

**The thesis overstates what the evidence shows.** The conclusion says closing
the optional-stopping gap "required a new algorithm". The paper's own ablation
records the prior-art changepoint mixture at PFA 0.000 on the identical
conformal front end. The gap was closed by the conformal layer plus any exact
e-process; the episode axis is a power claim, and its size is currently unknown.

**The experiment that isolates the contribution does not isolate it.** The two
arms are different classes over different betting grids (24-point vs 32-point),
and the ablation's own single-bet row shows a grid change alone produces an
effect the size of the claimed episode effect.

**Three numbers in the paper have no source.** The 0.555 windowed-objective
result appears nowhere in `results/`; the DKW figure 0.051 was computed at
n = 1500 and reported as n = 2000 (correct value 0.0441, ratio 11.9x not 14x);
the introduction quotes the eight-episode miss rates under a four-episode label.
Each contradicts §4.4's promise that "a stale number is not possible".

**Two tables lose columns off the page** in the compiled PDF, including the
miss-rate column of the paper's headline table, and Figure 4's caption asserts
the opposite of the paper's own finding.

## Standing

This is a manuscript whose problems are almost entirely of *scope and
reporting*, sitting on top of mathematics that four independent checks could not
break. That is the repairable kind. Every block names its remedy, and the
remedies are within reach of the existing harness. The revision is substantial
but it does not require new machinery, and the resulting paper would be
materially stronger than the one submitted.

The immutable revision roadmap is `review/phase3/revision_roadmap.md`.
