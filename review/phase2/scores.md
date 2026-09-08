# Phase 2 dimension scores (accumulating)

| Seat | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| perspective | — | — | — | **warn** | — | — |
| eic (journal-fit) | — | — | — | — | **block** | **block** (repairable) |
| da | — | — | **block** (repairable) | — | — | — |
| domain | — | **block** (repairable) | — | — | — | — |

## perspective (D4 = warn)
Trigger bound: "ethics, governance, or human-oversight implications are
acknowledged in a single perfunctory sentence without substantive treatment"

Findings: W1 edge claim is a same-CPU ratio, not transportable (major);
W2 compute table omits context-feature extraction and memory, grid size stated
inconsistently as 72/32/24 (moderate); W3 operator cannot distinguish a working
camera from a silently conservative one — miss rate is unobservable in the
field and the shift test is off in the default path (major); W4 no guidance for
alpha, calibration budget or recalibration trigger (moderate); W5 ethics
paragraph disclaims the question the deployment framing raises (major);
W6 four adjacent literatures unengaged — alarm management/human factors, signal
detection theory, SPC practice, public-sector algorithmic accountability
(moderate); W7 README misstates the headline cost by ~2.7x (minor).
Strengths: S1 abstention is a real operator-legible failure signature;
S2 the cross-domain generalisation is earned; S3 negative results reported.

### Author-side verification of the factual findings
- W7 CONFIRMED: README says 0.0006%; measured 0.606 us / 36.84 ms = 0.0016%.
- W2 CONFIRMED: paper states 72 grid points; EpisodePrior default is 24
  (6 kappa x 4 eta x 1 pi) after dilation was turned off by default.
