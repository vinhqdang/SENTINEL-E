# SENTINEL-E

**Sequential E-process NeTwork for INtermittent Litter/dumping Events** ---
anytime-valid sequential detection for streaming illegal-dumping surveillance.

Computer-vision systems for illegal dumping are built and evaluated as
fixed-sample-size classifiers: given a short clip, output one dumping /
no-dumping decision, report clip-level precision and recall. Deployment looks
nothing like that. A municipal camera runs for weeks, and what an operator cares
about is how many false alarms accumulate over the whole monitoring period,
because each one costs an enforcement dispatch. Applying a fixed threshold
repeatedly to a continuous stream and alerting the first time it is crossed is
optional stopping, and it inflates the true false-alarm rate by an amount that
clip-level evaluation never shows: in the experiments here, a per-frame rule
calibrated to a valid 1% level raises at least one false alarm on **100%** of
40-minute null streams.

SENTINEL-E turns any frozen per-frame detector into a network of anytime-valid
sequential detectors with time-uniform false-alarm control. It never touches the
backbone or the pixels --- it consumes one score per frame --- and its amortised
cost is about **0.0006%** of the backbone it wraps.

---

## The three layers

**Layer 1 --- conformal calibration** (`sentinel_e.conformal`, `sentinel_e.temporal`)

Raw scores become p-values that are super-uniform under the no-dumping null.
Three things make this work on real surveillance streams rather than on i.i.d.
draws:

* *Frame rate is not evidence rate.* Consecutive frames are near-duplicates. The
  decorrelation lag is estimated from the calibration autocorrelation, after
  projecting out the context-predictable drift, and one bet is placed per lag.
  Betting on every frame instead turns a nominal 1% into a realised 100%.
* *Conditioning, not pooling.* A night-time score compared against a
  daylight-dominated calibration set is marginally valid and conditionally
  useless. `ResidualConformalCalibrator` conformalises the context-normalised
  residual (clock, weather, ambient light, scene activity), which stays valid
  when the deployment context distribution shifts and scales to more context
  variables than binning can. `MondrianConformalCalibrator` is the exactly
  group-conditional alternative for one or two variables.
* *The calibration set is finite and gets reused.* Conditional on it, the
  p-values are only approximately uniform. `beta_calibration_levels` replaces
  each attainable level by its exact Beta order-statistic upper confidence
  bound, which is an order of magnitude tighter in the tail than the usual
  additive DKW inflation (0.0037 versus 0.051 at `n=2000`, `delta=1e-3`) and
  restores calibration-conditional validity.

Frames whose context falls outside the calibrated region are reported inactive
and bet on neutrally: a regime that was never calibrated cannot be certified.

**Layer 2 --- betting e-detector** (`sentinel_e.edetector`, `sentinel_e.betting`)

A changepoint prior `w_1, w_2, ...` and a betting function `f` define the wealth

```
W_t = sum_{j<=t} w_j prod_{s=j..t} f(p_s)  +  sum_{j>t} w_j
```

Keeping the *unreached* prior mass inside the statistic is what makes `W_t` an
exact non-negative supermartingale started at one, so Ville's inequality gives

```
P_infinity( exists t : W_t >= 1/alpha ) <= alpha
```

simultaneously over all frames. Dropping that mass recovers the classical
Shiryaev--Roberts statistic, which only admits an average-run-length bound. The
whole thing is the `O(1)` recursion `R_t = (R_{t-1} + w_t) f(p_t)`, evaluated in
log space; it also has a closed form that vectorises to a single cumulative
log-sum-exp for offline evaluation.

**Layer 3 --- the camera network** (`sentinel_e.graph`, `sentinel_e.gnn`, `sentinel_e.ebh`)

Offenders relocate, so a camera whose neighbours are accumulating evidence
should bet harder. A graph neural network supplies a per-camera hazard and stake
modulation from neighbour wealth. Because the modulation is *predictable* ---
measurable with respect to the past --- the wealth stays a supermartingale
whatever the network outputs, so the learned component cannot damage the
guarantee, only the delay. Fleet-wide decisions use e-BH, which controls the
false-discovery rate under **arbitrary** dependence, which matters because
adjacent cameras share weather and the graph couples them on purpose.

---

## Install and run

```bash
pip install numpy scipy matplotlib pandas torch     # torch only for the GNN layer
python -m pytest tests -q                           # 84 tests
python experiments/run_all.py --quick               # smoke test
python experiments/run_all.py                       # full protocol
```

Results land in `results/` (JSON), `results/figures/` and `results/tables/`.

### Minimal use

```python
from sentinel_e import SentinelE

model = SentinelE(alpha=0.01)                      # time-uniform false-alarm level
model.fit(calibration_scores, calibration_context) # confirmed no-dumping frames
result = model.run(scores, context)                # a monitoring stream
first_alarm = result.alarms.argmax() if result.alarms.any() else None
```

`context` is any per-frame array of observable covariates (time of day, weather
flag, ambient light, motion energy). `calibration_scores` are the backbone's
scores on confirmed no-dumping frames from the *same* camera.

### A camera fleet

```python
from sentinel_e.pipeline import FleetSentinelE
from sentinel_e.gnn import train_spatial_prior

model = train_spatial_prior(p_batches, adjacencies, statics, degrees, change_points)
det = FleetSentinelE.from_gnn(model, graph, alpha=0.01, fdr_level=0.1)
res = det.run(fleet)
res.ebh_rejected      # fleet-level decisions with FDR control
```

---

## Using your own data

SENTINEL-E consumes detector scores, not pixels, so any corpus reduces to the
same interface. Run your frozen backbone once, then:

```python
from sentinel_e.datasets import corpus_from_frame_table, write_scored_corpus
corpus = corpus_from_frame_table(frame_ids, clip_ids, scores, clip_labels)
write_scored_corpus(corpus, "data/scored/mivia_iwdd.npz")
```

`sentinel_e.streams.build_stream_from_clips` then splices those clips into long
monitoring streams: negative clips form the background, positive clips are
spliced in at random onsets, and calibration clips are held out so
exchangeability is preserved. This turns any clip-labelled video benchmark into
a streaming one without collecting new data.

Adapters and provenance notes are registered in `sentinel_e/datasets/adapters.py`
for Mivia-IWDD, TACO, ZeroWaste, UAVVaste, AerialWaste, MARIDA and MADOS. **No
adapter downloads or fabricates data.** Several corpora are behind request forms
and none of them ship detector scores, so a missing intermediate raises
`FileNotFoundError` with the exact recipe rather than silently substituting
something else. The simulator in `sentinel_e.streams` is the explicit,
clearly-labelled synthetic path.

---

## Repository layout

```
sentinel_e/
  conformal.py    Layer 1: split / weighted / Mondrian / residual calibration,
                  exact Beta calibration-conditional levels, support abstention
  temporal.py     decorrelation lag, detrending, FFT autocorrelation, Ljung-Box
  betting.py      power, linear, mixture and ONS-adaptive betting functions
  edetector.py    Layer 2: changepoint-mixture wealth process, Ville threshold
  graph.py        camera graph construction
  gnn.py          Layer 3: predictable spatial prior, offline GROW training
  ebh.py          e-BH and e-value merging
  baselines.py    fixed threshold, CUSUM, Shiryaev-Roberts, parametric
                  e-detector, E-SHIFT-style sub-Gaussian e-process
  streams.py      score-level stream simulator and clip-splicing protocol
  metrics.py      PFA, ARL0, detection delay, FDR and power
  pipeline.py     single-camera and fleet end-to-end pipelines
  datasets/       corpus representation and per-dataset adapters
experiments/      exp1 validity, exp2 delay-vs-FAR, exp3 shift, exp4 network,
                  exp5 compute, exp6 ablation
paper/            manuscript sources
tests/            unit tests
```

## License

Apache 2.0, see `LICENSE`.
