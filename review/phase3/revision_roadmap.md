# Revision Roadmap (immutable)

Frozen at panel close, 2026-09-08. Items are not renumbered or deleted; each is
resolved with a disposition (`done`, `superseded`, `declined + reason`).
Every item traces to a seat finding. Author-side directive folded in: reach
30+ pages through mathematics, algorithms, visualisation, data samples,
explanation and broader experiments — not padding. Groups E, F, G, H are where
that length comes from, and it is length the panel independently asked for.

Legend for source: **M**ethodology, **D**omain, **A**dvocate,
**P**erspective, **E**IC, **C**hair (synthesis no single seat made).

---
## A. Numbers with no source, or the wrong source (must fix first)
| # | Item | Source |
|---|---|---|
| A1 | Remove or generate the `0.555` windowed-objective result in §5.5. It is in no artefact and no script passes `credit_window`. Either run it and emit a macro, or state the outcome qualitatively. | M W3 |
| A2 | Correct the DKW comparison: at n=2000, delta=1e-3 the smallest attainable p-value is **0.0441**, not 0.051 (0.051 is the n=1500 value), and the improvement is **11.9x**, not 14x. Occurs at three places plus a contributions bullet. Make all four macro-generated. | M W9, D W6 |
| A3 | Fix §1.1: `\EpiMany*` binds to `counts[-1]` = **8** episodes; the sentence labels it four. | M W12 |
| A4 | Grid size: the shipped `EpisodePrior` default is **24** points (6 kappa x 4 eta x 1 pi). Paper says 72 in §4.2.3, 32 elsewhere. Bind to a macro. | M W15, E W4, P W2 |
| A5 | Table 8 caption says 40 fleets; the run used 30. `make_table` interpolates the module constant instead of the value used. Fix both caption and `run_all.py` defaults so the released numbers reproduce. | M W4, E W4 |
| A6 | README states the per-frame cost as 0.0006% of the frame budget; measured is 0.606 us / 36.84 ms = **0.0016%**. | P W7 |
| A7 | Abstract rounds 0.998 to "100%". State 99.8%. | E W4 |
| A8 | Three generated table captions cite wrong theorem numbers (Theorem 1 does not exist under the shared counter; the predictable-modulation result is Theorem 4, cited as Theorem 2). Give theorems their own counter. | E W3, M W4 |
| A9 | `shin2024edetectors` DOI resolves to an unrelated paper: correct to 10.51387/23-NEJSDS51. Fix `bashkirova2022zerowaste` pages to 21115--21125. | D W7, D W4 |

## B. Report the uncertainty that already exists
| # | Item | Source |
|---|---|---|
| B1 | Print `pfa_ci` in Table 8. At 200 replicates every PFA 0.000 row has Wilson interval [0.0000, 0.0188] — 1.9x nominal — so **no** validity verdict in the ablation is resolvable. Replace the check-mark column, and the "5 of 19 fail" count, with something the design can support. | M W1 |
| B2 | The abstract's x2.0 shift breaking point has stored interval [0.0078, 0.0503], which **contains nominal 0.01**; x1.5 likewise. The measured breaking point is x3.0. Restate abstract, §5.4 and Limitations. | M W2 |
| B3 | Put intervals on every head-to-head episode difference (Tables 11, 12). | M W13, A C4 |
| B4 | `EpiBestCount` is an argmax over the sweep quoted as the headline. Pre-specify the operating point or report the whole sweep with multiplicity acknowledged. Drop "the gap widens": the relative reductions run 14.2%, 21.5%, 9.3%. | M W13 |
| B5 | Table 12's pi=0.35 row is byte-identical to Table 11's 4-episode row; the other three rows separate nothing. Say so or drop the table as independent evidence. | M W14 |
| B6 | §6.2/6.4/6.7 reprint the same 200 seeded runs (`seed0 = 10_000`, `add = 5140.040609137056` recurs across four JSONs). Stop presenting the repetition as corroboration. | A M4 |

## C. Make the isolation experiment isolate  *(exp9, harness built and running)*
| # | Item | Source |
|---|---|---|
| C1 | Re-run episodic vs changepoint with **one class over one kappa grid**, differing only in the eta axis: `eta=(0,)` (Proposition 3 boundary, = the changepoint mixture) vs the shipped eta grid. Adds a grid-size-matched control (`cp24`) separating "more mixture components" from "an episode-end axis". | A C3 |
| C2 | Report **paired** differences: both arms share seeds. Bootstrap CI on censored delay, exact McNemar on discordant miss pairs. This is more powerful than the unpaired comparison and is the right analysis, not a concession. | A C4, M W13, C |
| C3 | **Chair item, no single seat made it.** The only properly powered validity experiment is Experiment 1 (1500 reps). It runs the episodic arm alone. Run the changepoint mixture through it. Until then the claim that only the new algorithm holds the level is unsupported at any replication count. | C |
| C4 | Put the changepoint mixture into Table 3 and Figure 2. It is announced as a baseline at §5 and appears in neither, which is where the abstract's uniqueness claim is established. | A C2 |
| C5 | Sweep the episode comparison across **realised** false-alarm rates. Every comparison currently sits at PFA 0.000 against nominal 0.01, while the paper advises deploying at an empirically calibrated threshold. | A M5 |
| C6 | Add a genuine persistent-change arm (eta identically 0 in the data-generating process). The 1-episode row is an 800-frame episode in a 90,000-frame stream, not a persistent change, so the "boundary case" is asserted rather than tested. | A part(a) |

## D. Scope the claims to the evidence
| # | Item | Source |
|---|---|---|
| D1 | Restate the thesis. "Closing that gap required a new algorithm" is negated by the paper's own ablation (changepoint mixture, PFA 0.000, identical front end). The entailed thesis: a conformal front end plus an anytime-valid stopping rule closes the optional-stopping gap; the episode axis is a cheap prior that may improve power where events recur. | A C1 |
| D2 | Re-scope novelty to "new **within the anytime-valid e-process framework**". What the classical transient-change literature lacks is an exact non-negative supermartingale with time-uniform Ville control and an O(1) forward recursion; say that, and cite what exists. | D W1 |
| D3 | Reconcile the motivating phenomenology with the regime that produces the advantage. The abstract says episodes recur "weeks later"; the gain lives at 4-8 episodes inside one 60-minute stream. In the 1-episode row the episodic mixture is nominally worse on both quantities. State the deployment-relevant quantity: episodes elapsing before the first alarm within one monitoring run. | A C5 |
| D4 | Propagate the two negative results into the contributions list. Bullet 2 (the episode posterior as inter-camera message) is justified by the graph layer §6.5 then refutes. | A M3 |
| D5 | Fix the ablation prose: §5.7 lists raw (unthinned) calibration as validity-breaking, which Table 8 marks **valid**; and omits Mondrian conformal, which the table marks invalid. §4.1 repeats the error. | M W11 |
| D6 | Figure 4's caption says "the learned spatial prior shortens delay" — the opposite of the paper's finding. | E W2 |
| D7 | Correct the IWDD contest characterisation: **ten** teams, not the five enumerated; the winner (AGH-EVS, F1 0.57) and the organisers' overview are uncited; the official protocol is P/R/F1 **plus normalised notification delay, frame rate and peak memory**; §6.2 of the overview analyses the field in ROC space with an explicit FPR axis. The substantive claim survives with the qualifier "over a continuous monitoring horizon". | D W2 |
| D8 | Correct the dataset chronology: the contest ran on Mivia-IWDD (400 clips, 200/200) with a private 100-clip test set; Mivia-IWDD-500 postdates the workshop by seven months. Fix the §3 protocol advice, which assumes 250 negatives. | D W3 |
| D9 | ZeroWaste is an industrial waste-sorting corpus, not in-the-wild litter — the repo's own adapter says so. Fix the prose to match the code. | D W4 |
| D10 | E-SHIFT is **sub-exponential**, not sub-Gaussian; it assumes i.i.d. (stronger than exchangeability) and **does** provide an alpha-mixing extension for correlated streams, which we omit while making autocorrelation our centrepiece. | D W5 |
| D11 | The edge-deployment claim is a same-CPU ratio measured against a stand-in backbone, not a transportable statement. Say what was measured. | P W1, E W10 |
| D12 | The tested horizon is 2.67 h; the motivating claim is "weeks". Either test longer or scope the claim. | E W8 |

## E. Related work the paper must engage  *(new section; primary sources verified by the chair)*
| # | Item | Source |
|---|---|---|
| E1 | **Transient / intermittent / epidemic change detection.** Guepie, Fillatre & Nikiforov, *Sequential Analysis* 31(4):528-547, 2012, and IEEE TIT 2017; Sokolov, Spivak & Tartakovsky, *Sequential Analysis* 42(3):269-302, 2023 (arXiv:2210.17342); Tartakovsky et al., arXiv:2102.01310; Fuh & Mei, IEEE TSP 63(18), 2015; the epidemic-changepoint line (Levin & Kline 1985; Fisch, Eckley & Fearnhead, CAPA; Juodakis & Marsland arXiv:2008.08240 — the DA mis-attributed the last of these, correct on the way in). | D W1, A M2 |
| E2 | **Fixed-share / switching-expert mixtures.** Herbster & Warmuth, *Machine Learning* 32(2):151-178, 1998; the switch distribution (van Erven, Grunwald & de Rooij); Koolen & de Rooij's HMM formulation; and arXiv:2408.14073, score-based change detection via tracking the best of infinitely many experts — the bridge itself. Equation 4 is fixed-share with two experts, one neutral; say so and say what the e-process framing adds. | A M2 |
| E3 | **Conformal test martingales for change detection.** `vovk2003exchangeability` is in `refs.bib`, uncited. It and its successors are the closest existing Layer 1 / Layer 2 architecture and must be named and distinguished in §Related work. Four other bib entries are also uncited. | D W8 |
| E4 | **Four adjacent literatures for the deployment framing**: alarm management and human factors, signal detection theory, statistical process control practice, public-sector algorithmic accountability. | P W6 |
| E5 | Note that classical sequential change detection **has** reached pollution surveillance (Guepie et al., IFAC 2012, water-distribution contamination). The accurate gap is that the *anytime-valid* branch has not. | D S2 |

## F. Experiments to add
| # | Item | Source |
|---|---|---|
| F1 | **Per-episode detection metric.** `metrics.detection_delay` scans from the first onset and `miss_rate = misses / n_signal` is stream-level, so a run that sleeps through episodes 1-7 and fires on episode 8 scores a clean detection. The claimed accumulate-across-episodes mechanism is currently unmeasured. Add per-segment detection and per-episode delay, or drop the mechanism claim. | A M1 |
| F2 | Real per-frame backbone scores over Mivia-IWDD. The EIC scores D6 `block` because the contribution is domain-detachable as executed, and names this as the one experiment that repairs it: "requires only that a backbone be run once over the corpus." This is the single highest-value item in the roadmap. | E W6, W7 |
| F3 | Vary the simulator configuration. All headline experiments run at `n_episodes=4, event_length=800, episode_gap=6000, event_intermittency=0.35`, which departs from the library defaults in the three directions the exp8 sweep shows favour the method. Report a grid, not a point. | A M4 |
| F4 | Compute budget: the table omits context-feature extraction and memory. Add both. | P W2 |
| F5 | Operator-observability experiment. A field operator cannot distinguish a working camera from a silently conservative one: miss rate is unobservable and the shift test is off in the default path. Give a diagnostic that is computable from what an operator has. | P W3 |

## G. Presentation
| # | Item | Source |
|---|---|---|
| G1 | Tables 5 and 12 lose columns off the page (Overfull \hbox 98.6pt / 77.4pt / 114.4pt); the miss-rate values on the headline table are invisible in the compiled PDF. | E W1 |
| G2 | Move to the Springer submission form. | E W5 |
| G3 | Table 10 is described as "checkable from the calibration set alone" but is computed from 191,890 deployment betting instants on labelled null streams, which an operator does not have. | M W5 |
| G4 | Substantive treatment of ethics, governance and human oversight; and deployment guidance for alpha, the calibration budget, and when to recalibrate. Currently one perfunctory paragraph and no guidance. | P W4, W5 |

## H. Theory to add or repair  *(no theorem withdrawn)*
| # | Item | Source |
|---|---|---|
| H1 | **Add the composition corollary.** Theorem 1 gives conditional super-uniformity on an event of probability >= 1-delta; Theorem 2 assumes it exactly. The end-to-end guarantee is **alpha + delta**, which the library already documents (`pipeline.py:68`) and the manuscript never states. At delta = 1e-3 this is 10-20% of the reported alphas. | M W6 |
| H2 | **Define the filtration for the fleet, and state the joint assumption.** Camera i's hazard is produced by message passing over neighbours' posteriors, so rho_{i,t} is measurable w.r.t. the *joint* history; the supermartingale step then needs conditional super-uniformity given the joint past — stronger than the per-camera condition, and precisely what the paper says the deliberate couplings violate. e-BH repairs the combination step, not the per-camera e-value. Either state the joint assumption or restrict the controller to each camera's own history. | M W7 |
| H3 | Repair Theorem 4's freezing step: `f_{kappa_eff}` is nonlinear in `kappa_eff`, so nothing is "taken outside" the conditional expectation. The conclusion is true; the argument needs a regular conditional distribution and joint measurability of (k, p) -> f_k(p). | M W7 |
| H4 | Theorem 1 admits an arbitrary positive budget, for which the Beta levels need not be monotone; monotonicity is asserted in the proof. Either add the (j+1)^-2 allocation as a hypothesis or take the running maximum in the statement, as the code already does defensively. | M W8 |
| H5 | The Kish effective-sample-size substitution under likelihood-ratio weights is an approximation the code flags (`weighted_bound='effective_beta'`, "exact only for uniform weights") and the manuscript presents without qualification. Its empirical support is a detector with miss rate 0.98-1.00 — a detector that never fires cannot evidence a calibration correction. Derive the weighted bound or state that its validity is asserted. | M W16 |
| H6 | Distinguish Shiryaev's geometric-prior statistic from the Shiryaev--Roberts sum with unit weights (the rho -> 0 limit after rescaling). The paper keeps `shiryaev1963` and `roberts1966` distinct elsewhere. | M S2 |
| H7 | Reconcile the conditional-independence diagnostic with Theorem 2's hypothesis. Median Ljung--Box p = 0.005 and only **18.3%** of cameras pass at the 5% level after thinning; the caption says the data are "consistent with" the premise. The test does not fail to certify, it rejects, on four cameras in five. The operational claim survives via Experiment 1; the claim that Theorem 2's hypothesis holds for the deployed procedure does not. | M W5 |
