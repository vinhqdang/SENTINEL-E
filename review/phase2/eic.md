Analysis complete. Here is my report.

---

contract_role: eic

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: block
trigger: "or a figure/table carrying a headline claim that lacks the labelling, units or scale needed to read it"

### D6: venue_fit_and_contribution
score: block
block_class: repairable
trigger: "The actual contribution is generic to its methodological subfield with the surveillance setting supplying only motivation, framing or nomenclature, so that removing the application domain would leave the contribution intact"

## Review Body

I write as the handling editor for this collection, and I want to be clear at the outset about what kind of paper this is and what kind of problem that creates. This is an unusually disciplined manuscript. It states its claim rather than its topic, it delimits its prior art before a reviewer has to, it reports two results that went against the authors, and its limitations section says the hard thing first. Those are not common virtues and I do not want the two blocks below read as a judgement on the author's care. They are a judgement on two different things: a compiled artefact in which the paper's two most load-bearing tables physically lose columns off the page, and a fit question that I do not think the manuscript, as pitched, answers in this collection's favour.

On D6 I should say plainly what I am and am not deciding. I am not deciding that the work is unpublishable, that the theorems are wrong (not my seat), or that the domain framing is insincere. I am deciding that the centre of gravity of this manuscript is a supermartingale construction whose empirical support comes entirely from a generative model the author wrote, and that this collection asked for robustness across diverse real environments, datasets and benchmarks, and deployment on constrained hardware in real time. On the collection's own terms — video analysis for waste tracking, CV for detecting illegal disposal, remote sensing and UAV monitoring, datasets and benchmarks, deployment case studies — the manuscript as executed matches none of the five. It processes no video, trains no detector, touches no corpus, and reports no deployment. What it contributes to this readership is an argument about evaluation protocol, which is real and which I credit below, wrapped around a method whose validation lives at one remove from every artefact this collection's readers work with.

I score that block rather than fatal, and repairable rather than not, deliberately. The manuscript names its own remedy and says it is cheap: run the protocol on real per-frame backbone scores over Mivia-IWDD, which "requires only that a backbone be run once over the corpus." If that is true, the fit problem is one experiment away from being fixed, and the resulting paper would be a strong fit rather than a marginal one. That is a major revision, not a rejection.

### S1: The protocol critique is genuinely this collection's business
**Evidence Anchor**: text: "is a false-alarm rate over a monitoring horizon, an average run length, or a detection delay under continuous monitoring"
**Confidence**: 5 — as handling editor I read the IWDD contest entries when scoping this collection, and the characterisation of their reporting practice is accurate. The observation that clip-level precision and recall cannot express what a municipality actually manages is the one part of this manuscript that could not have been written by someone outside the domain, and it is the part I would most want this readership to see.

### S2: Self-delimitation of the contribution is exemplary
**Evidence Anchor**: text: "The changepoint mixture with retained tail mass that our construction reduces to is not new"
**Confidence**: 5 — editorial judgement of a paragraph placed before the related-work section rather than buried in it. An explicit "what we do not claim" paragraph that concedes the reduction, the conformal layer and the e-BH layer as prior art is rarer than it should be, and it made this review faster and more accurate. It also, as I note in W4 below, sets the bar the residual contribution must clear.

### S3: The conditional-versus-censored delay correction is portable
**Evidence Anchor**: text: "detection delay conditioned on detection is averaged over a different subset of runs for each method"
**Confidence**: 4 — I can verify from the tables that the two orderings do diverge (Table 5: E-SHIFT is second-fastest on conditional delay at a 0.69 miss rate, sixth on censored delay), and the reasoning is elementary enough for me to check without the methodology seat. This is a measurement point that would improve comparisons across this whole collection.

### S4: Candour about the evidence base
**Evidence Anchor**: text: "This is the main limitation and we do not want it buried."
**Confidence**: 5 — direct reading. The limitations section leads with the simulation, states what the simulation licenses and what it does not, and reports that the dataset adapters raise rather than substitute. I want to record this because my D6 block is about the consequences of the simulated basis, not about any attempt to conceal it.

### W1: Two headline tables lose columns off the page in the compiled artefact
**Severity**: major
**Evidence Anchor**: absence: Table 5 (p. 15) and Table 12 (p. 21) of sentinel_e.pdf — expected the "miss rate" and "lag" column values inside the text block; checked the compiled PDF pages 15 and 21 at 100 dpi, /home/user/SENTINEL-E/results/tables/tab3_delay_far.tex, /home/user/SENTINEL-E/results/tables/tab8_ablation.tex, and /home/user/SENTINEL-E/paper/sentinel_e.log (Overfull \hbox of 98.6pt, 77.4pt and 114.4pt)
**Confidence**: 5 — I rendered both pages. On Table 5, which the manuscript itself calls the comparison the paper exists to make, the header reads "miss" at the page edge and not one miss-rate value is visible; the prose immediately below quotes 0.01 from that invisible column. On Table 12, the ablation, both the miss and lag columns are cut to "m" and "0.", and the ablation's stated point — that validity failures cluster in the temporal and calibration components while the rest move only delay — cannot be checked, because the delay/miss pairing is half off the page. The manuscript is currently set in the generic article class at a 2.5 cm margin; Springer's measure is narrower, so this gets worse rather than better on conversion. The fix is mechanical (a smaller font, a rotated float, or a split table), which is why I flag it as repairable presentation rather than a defect of the work. But as the document stands, a referee cannot read the numbers the argument rests on, and that is the trigger I committed to in Phase 1.

### W2: A figure caption asserts the opposite of the result it illustrates
**Severity**: major
**Evidence Anchor**: text: "The learned spatial prior shortens delay"
**Confidence**: 5 — the caption of Figure 4 says this; Section 6.5, Table 8 and the abstract all say the learned prior lengthens delay (215.4 s against 204.6 s) and cuts power (0.704 against 0.816), and the conclusion lists it among the components that "did not survive contact with measurement." A reader who skims the figures — which is most readers — takes away the reverse of the paper's own finding on the one component the paper is most careful to be honest about. This is precisely the caption that most needed to be right, since it is the visual record of a disclosed negative result. Related, and worth fixing in the same pass: panel (c) of that figure plots three bars all at zero against a dashed target line, so it conveys nothing a sentence would not, and panel (b) carries no error bars where panel (a) does.

### W3: Three generated table captions cite the wrong theorem
**Severity**: moderate
**Evidence Anchor**: text: "tail resolution the exact Beta levels of Theorem 1 provide"
**Confidence**: 5 — verified against the compiled numbering. Theorem 1 does not exist in the document; the counter is shared, so item 1 is Definition 1 (time-uniform false-alarm control), the Beta result is Theorem 2, the episodic supermartingale is Theorem 3 and the predictability result is Theorem 5. Table 11's caption points the Beta claim at Theorem 1; Table 10's caption points the conditional-independence premise at Theorem 1 when it means Theorem 3; Table 8's caption credits the predictable-modulation guarantee to Theorem 2 when it means Theorem 5. The body prose gets all of these right, which localises the cause: the captions are hard-coded strings in the table-generating scripts rather than \cref calls, so they were never resolved against the document. Every formal cross-reference a reader can follow from a table is currently wrong.

### W4: Configuration numbers in the prose match no artefact, and the abstract's opening figure disagrees with its own table
**Severity**: major
**Evidence Anchor**: text: "With 72 grid points the per-frame cost is a few hundred floating-point"
**Confidence**: 4 — I traced the grid to source. The default EpisodePrior in /home/user/SENTINEL-E/sentinel_e/episodic.py is six kappa values by four eta values by one pi value, which is 24 points, and that is what experiments/runner.py instantiates for the episodic detector; the compute table reports a "32-point mixture", which is one rung of the 8/16/32/64 sweep in results/exp5_compute.json. Seventy-two is neither. Since the paper's parameter-free claim and its edge-cost claim both rest on the size of that grid, a reader cannot reconcile the method description with the measurement. Two further contradictions of the same kind: the caption of Table 8 states 40 held-out fleets with events and 40 all-null fleets, while the Figure 4 caption, the body text and results/exp4_network.json all say 30; and the abstract's opening empirical claim, that a valid per-frame rule false-alarms on 100% of 40-minute null streams, is a zero-decimal rounding of the 0.998 printed in Table 1 four pages later. Each is small; together they undercut the manuscript's own strongest procedural claim, that every quoted number is macro-generated so "a stale number is not possible."

### W5: The manuscript is not in the venue's submission form
**Severity**: minor
**Evidence Anchor**: text: "For submission, replace the preamble with Springer's sn-jnl.cls; the body is class-independent."
**Confidence**: 5 — the preamble says so itself. Beyond the class, the file still carries the draft scaffolding that renders a missing table or figure as a visible "results pending" box, and the bibliography is plainnat author-year rather than the Springer style. The reference list also shows casing damage from unbraced bib titles ("the dvoretzky–kiefer–wolfowitz inequality"). None of this bears on the science and I would normally leave it to production, but the class change interacts with W1: converting to a narrower measure without addressing the wide tables will make the truncation worse.

### W6: The contribution is domain-detachable, and the manuscript says so
**Severity**: critical
**Evidence Anchor**: text: "the argument nor the algorithm is specific to dumping"
**Confidence**: 5 — this is the fit judgement that is squarely my seat, and the manuscript hands me the test. Apply it: strip illegal dumping from this paper and the episodic e-process, the two-state forward recursion, Proposition 1's reduction to the changepoint mixture, the exact Beta calibration levels, the predictable-modulation theorem and the e-BH fleet layer all survive untouched. What is left behind is the motivation and the nomenclature. The conclusion offers wildlife poaching, illegal fishing, gas leaks and infrastructure faults as equally good homes, which I read as an accurate self-description rather than an overreach. A paper whose method operates on one scalar per frame, whose data are drawn from a parametric score model, and whose theorems are statements about non-negative supermartingales has its natural readership at a sequential-analysis or statistical-ML venue, and would be well received there. What makes this a block rather than a rejection is that the domain-specific half of the argument — that the IWDD evaluation protocol cannot express the quantity a municipality manages — is real, is not detachable, and is exactly what this collection's readers need to hear. That half is currently the smaller half.

### W7: No collection gap is closed empirically
**Severity**: critical
**Evidence Anchor**: absence: Sections 5 and 6 and the results directory — expected at least one experiment on real per-frame detector scores, from Mivia-IWDD-500 or any operational camera; checked paper Sections 5–6, results/exp1_validity.json through results/exp8_episodes.json, experiments/*.py, and all thirteen files in results/tables/
**Confidence**: 5 — every one of the eight experiment payloads is generated by the score-level simulator; there is no dataset adapter output anywhere in the artefacts. Measured against this collection's three stated gaps: robustness in diverse environments is addressed only as robustness to a shift the author's own simulator injects into a covariate the author's own simulator defines; the datasets-and-benchmarks gap is addressed by proposing a clip-splicing protocol and then not running it, so the collection gains a recipe rather than a benchmark; and real-system deployment on constrained hardware is addressed in W10 below. The author's honesty about this is genuine and I have credited it as S4, but honest scoping does not convert simulated evidence into applied evidence. The concrete remedy is the one the manuscript already names: run the identical protocol on real backbone scores over Mivia-IWDD-500. If build_stream_from_clips works as described, that is a single compute job, and it would move this dimension from block to pass in one revision cycle.

### W8: The horizon that motivates the paper is two orders of magnitude beyond the horizon tested
**Severity**: major
**Evidence Anchor**: text: "A municipal camera runs for weeks"
**Confidence**: 4 — arithmetic from the tables. The framing throughout is weeks and months of continuous monitoring; the whole force of the time-uniform argument is that per-frame guarantees decay with the horizon. The longest horizon actually swept is 240,000 frames, 2.67 hours, in Table 2; every other experiment runs at 60,000 or 90,000 frames, that is 40 or 60 minutes. The horizon experiment is the one place where a longer sweep costs only compute, and it is the experiment whose extension would most directly support the paper's central rhetorical claim. As it stands the manuscript demonstrates the failure of per-frame rules over hours and asserts it over months, and asks the reader to supply the extrapolation that the theory, but not the measurement, licenses. This is a scoping mismatch rather than an error, and it is the cheapest of the D6 repairs.

### W9: The residual novelty, after the author's own concessions, is narrow for this readership
**Severity**: major
**Evidence Anchor**: text: "contribution is the episode axis on top of it, and the O(1) recursion that axis happens to make"
**Confidence**: 4 — this follows from the manuscript's own accounting rather than from any independent claim of mine, which is why I hold it at 4 rather than 5; whether the episode axis is novel against the wider sequential-testing literature is the domain seat's call, not mine. Take the author's ledger at face value. The changepoint core reduces to a published PFA-controlled e-detector and is disclaimed. The conformal and e-BH layers are existing machinery, credited. The dilated emission is withdrawn as a negative result. The learned spatial prior is reported as not earning its place. What remains for this collection's readers is the episode-end axis plus the Beta-level substitution, and the measured payoff of the episode axis on the author's own streams is a censored delay of 245 s against 311 s and a miss rate of 0.015 against 0.055 at four episodes per camera, turning negative at one episode. That is an honest and clearly-presented delta. It is also a modest one, obtained in simulation, on an axis whose practical relevance depends on an empirical premise about dumping recurrence rates that the paper asserts ("a municipality knows the answer") rather than establishes from any corpus. For a statistics venue this is a clean incremental result well told. For a topical collection whose readers are looking for something to deploy, it is thin.

### W10: The edge-deployment claim is measured against a stand-in
**Severity**: moderate
**Evidence Anchor**: text: "it stands in for a deployed detector rather than reproducing a specific published one"
**Confidence**: 4 — from the Table 9 footnote and results/exp5_compute.json. The headline "0.0016% of the backbone" is a ratio whose denominator is a synthetic depthwise-separable stack the author timed on the same general-purpose CPU that ran the statistical experiments, not any of the five IWDD backbones the paper positions against, and not on edge hardware. The numerator is measured at a 32-point mixture and a 32-camera fleet, neither of which is the configuration described in the method (see W4). I do not doubt that an O(1) two-state recursion is arithmetically negligible next to any video backbone — the claim is almost certainly true. But this collection specifically asked for solutions demonstrated under limited computational resources in real time, and a ratio against a hypothetical denominator on a workstation is not that demonstration. Timing the wrapper alongside one of the actual contest backbones on a representative edge device would convert a plausible assertion into the deployment evidence the call asked for, and it is a day's work.