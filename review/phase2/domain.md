I have completed verification. Here is my report.

---

contract_role: domain

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: block
trigger: "a claimed literature gap that is occupied by work the manuscript either omits or dismisses on grounds its own cited evidence does not support, such that the novelty framing depends on the misstatement"
block_class: repairable

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

I checked every verifiable domain assertion in this manuscript against primary sources: the Image and Vision Computing dataset paper, the MIVIA contest site, all seven published WACV 2026 WasteVision IWDD papers plus the organisers' contest overview, the arXiv records for Saha–Ramdas and Khan–Syed, and Crossref records for the six auxiliary corpora. The result is mixed in an unusual way. The manuscript's *empirical* domain facts are, with two exceptions, correct — unusually so for a paper of this framing. Its *negative* claims about the literature are where it fails, and one of them is load-bearing enough to block.

The verdict on the two claims the manuscript most depends on: the surveillance-side gap (no IWDD entry reports a false-alarm rate over a monitoring horizon) is **true**, and the anytime-valid gap (no e-process/conformal-martingale machinery applied to waste, dumping or pollution surveillance) is **also true** — I searched hard for counterexamples and found none. But the *change-detection*-side claim — that no sequential detector models an episode that ends — is **false**, and it is the one the paper calls its main contribution's axis of novelty.

### W1: The "episode duration is an axis the literature fixed at infinity" claim is refuted by the transient-change-detection literature, which the manuscript never cites

The manuscript's main contribution is framed as a strict generalisation along an axis it asserts is unoccupied: that every sequential change detector models a permanent switch, and that no procedure models a change of finite duration, still less a recurring or intermittent one. This is not correct. There is a mature sub-literature on exactly this object. Guépié, Fillatre and Nikiforov, "Sequential Detection of Transient Changes," *Sequential Analysis* 31(4), 2012 (doi:10.1080/07474946.2012.719443) treats a change of finite duration that must be detected *before it disappears*, with a window-limited CUSUM and a criterion minimising worst-case missed detection under a false-alarm constraint; the same authors extend it in "Detecting a Suddenly Arriving Dynamic Profile of Finite Duration," *IEEE Trans. Inf. Theory*, 2017 (doi:10.1109/TIT.2017.2679057). The Bayesian formulation of transient quickest detection models the change as arriving at Γ₁ and *departing* at Γ₂, with explicit pre-change / in-change / out-of-change states — structurally the manuscript's two-state chain. Fuh and Mei, "Quickest Change Detection and Kullback–Leibler Divergence for Two-State Hidden Markov Models," *IEEE Trans. Signal Processing* 63(18), 2015 (doi:10.1109/TSP.2015.2447506) is a two-state HMM quickest-detection treatment. None of these appears in the manuscript or in `refs.bib`, and no distinction is drawn against them.

I want to be precise about what this does and does not destroy, because I do not think it destroys the algorithm. What I could not find in that literature is an *e-process* formulation: a construction that is an exact non-negative supermartingale over episode trajectories, gives time-uniform Ville control rather than an asymptotic ARL/PFA statement, and collapses to an O(1) HMM forward recursion. The transient-change literature is CUSUM-, SPRT- and POMDP-shaped, with ARL-constrained optimality. So the defensible novelty claim is "the episode axis is new *within the anytime-valid e-process framework*", not "the change-detection literature has held it fixed at infinity". As written the manuscript claims the latter, in the abstract, the introduction, the contributions list and §Related work, and the claim is wrong each time. The repair is to cite the transient-change line, state what it does, and re-scope the novelty — which leaves \cref{thm:episodic} and \cref{prop:generalisation} untouched. That is why I score this repairable rather than fatal. It is nonetheless a block: a reviewer from the change-detection side will recognise the omission immediately, and the paper's headline positioning currently rests on it.

**Severity**: Blocking (repairable)
**Evidence Anchor**: text: "none of them models an episode that ends, let alone one that recurs or that is internally intermittent"
**Confidence**: 5 — I hold the primary records (Crossref DOIs, abstracts) for all three counterexample lines, and I have worked with the transient-change literature in a surveillance context; the only judgement call is the scope of the repair, not the existence of the prior work.

### W2: "Every entry in the 2026 IWDD contest" is asserted over an enumeration of five of ten teams, and omits the organisers' own overview paper

The universal quantifier does the work in this paper's motivation, so the enumeration behind it has to be complete. It is not. The contest overview — Bouwmans, Greco, Piérard, Ricciardi, Sansone, Van Droogenbroeck, Vento, "Illegal waste dumping detection," WACV 2026 WasteVision — states that **ten teams** participated. The manuscript names five: ALEXKNIGHTS (Almorsy), SAFE:CAUTION (Kim), GARBERUS (Scarrica), ARRAY (Kairanbay) and SUDOPI (Zbuce). Two further contest entries were published in the same proceedings and are not cited: IMSLAB — Bhat and Lin, "Boundary-Sensitive Start-Time Estimation with Onset-centric Temporal Detection for Illegal Waste Dumping in Surveillance Video" (doi:10.1109/WACVW68408.2026.00072), and UNICASS — Delussu and Putzu, "A Lightweight Temporal Detection Framework for Illegal Waste Dumping in Real Surveillance Footage." The contest *winner*, AGH-EVS (F1 = 0.57, the best of the field), is not mentioned at all, and neither is the organisers' overview paper.

I checked all seven published entries. The substantive claim survives: none reports a false-alarm rate over a monitoring horizon, an ARL, or any anytime-valid quantity. But two corrections are needed. First, the contest's official protocol is precision / recall / F1 *plus normalised notification delay* D_norm, processing frame rate and peak memory — not "F1 and a timestamp error"; several entries (Zbuce, Bhat, Delussu–Putzu) report D_norm, a delay measure, rather than a timestamp error, and Scarrica reports only P/R/macro-F1 within a tolerance window. Second, §6.2 of the overview paper analyses the field in **ROC space with an explicit False Positive Rate axis**. That is still per-clip specificity, so the manuscript's distinction holds, but the claim in §\ref{sec:delay} that "no existing waste-surveillance paper reports its analogue" needs the qualifier "over a continuous monitoring horizon" attached, since the organisers do pair a delay measure with an FPR.

**Severity**: Major
**Evidence Anchor**: text: "Every entry in the 2026 IWDD contest on the Mivia-IWDD benchmark follows this protocol"
**Confidence**: 5 — I downloaded and read all seven WasteVision IWDD papers and the contest overview, and confirmed the team-to-paper mapping from the overview's own reference list.

### W3: The contest did not run on Mivia-IWDD-500, and the 500-clip release postdates it

The dataset facts themselves are right (see S1), but the causal story around them is not. The 2026 IWDD contest was run on **Mivia-IWDD: 400 videos, 200 positive / 200 negative**, released as the training set, with results reported on a **private, unreleased 100-video test set** (50 positive / 50 negative) collected in scenarios disjoint from training. Mivia-IWDD-500 (250/250) is the larger release described in the *Image and Vision Computing* paper, which Crossref dates to volume 174, article 106125, **October 2026** — seven months *after* the March 2026 WACV workshop. So the contest did not build on Mivia-IWDD-500; if anything the ordering runs the other way.

This is not pedantry, because it propagates into the paper's own recommended protocol. §\ref{sec:problem} ("From clips to streams") tells a reader to reserve a disjoint subset of confirmed-negative clips as $\cD$ and concatenate the rest into a background stream. On the public release that means partitioning 200 negatives, not 250; and the clips on which every number in the contest was reported are not obtainable at all. Given that the paper's stated first priority for future work is running the protocol on Mivia-IWDD, the reader needs the correct inventory.

**Severity**: Moderate
**Evidence Anchor**: text: "and the 2026 IWDD contest built on it attracted a strong and varied field"
**Confidence**: 5 — the contest site states 400/200/200 and the private 100-clip test set explicitly; the overview paper repeats both; the IVC issue date is from Crossref.

### W4: ZeroWaste is characterised as an in-the-wild litter dataset; it is an industrial waste-sorting corpus, and the paper's own adapter says so

ZeroWaste (Bashkirova et al., CVPR 2022) is a materials-recovery-facility conveyor-belt dataset for deformable-object detection and segmentation in extreme clutter — the authors describe it as "the first in-the-wild **industrial-grade** waste detection and segmentation dataset." Grouping it with TACO under "image-level litter detection" conflates two quite different acquisition regimes and label semantics, and a domain reader will notice, because the paper's own `sentinel_e/datasets/adapters.py` gets it right ("cluttered waste-sorting imagery and video frames", "heavy-clutter robustness check"). Fix the prose to match the code. Separately, the `bashkirova2022zerowaste` page range is wrong: the CVPR 2022 proceedings give 21115–21125, not 21147–21157.

The other five corpora check out exactly. TACO (Proença and Simões, arXiv:2003.06975, 2020) is correctly given as a preprint. UAVVaste is correctly attributed to Kraft, Piechocki, Ptak and Walas, *Remote Sensing* 13(5):965, 2021. AerialWaste to Torres and Fraternali, *Scientific Data* 10(1):63, 2023. MARIDA to Kikaki et al., *PLOS ONE* 17(1):e0262247, 2022, and MADOS to Kikaki, Kakogeorgiou, Hoteit and Karantzalos, *ISPRS J. Photogramm. Remote Sens.* 210:39–54, 2024 — both correctly identified as Sentinel-2 marine-pollution benchmarks. Modality, venue, year and authorship are right in every one of those five.

**Severity**: Minor
**Evidence Anchor**: text: "Image-level litter detection is well served by TACO"
**Confidence**: 4 — Crossref and the BU project page are unambiguous on ZeroWaste's domain and pagination; the severity judgement (loose grouping rather than a false claim) is mine.

### W5: E-SHIFT is described as a sub-Gaussian e-process; it is sub-exponential, and its mixing extension goes unmentioned

The E-SHIFT citation itself is sound, and I want to record that explicitly because the arXiv metadata is misleading: arXiv:2510.03839's *record* title is "Technical note on Sequential Test-Time Adaptation via Martingale-Driven Fisher Prompting", but the v2 document (29 May 2026) is indeed titled "E-SHIFT: Anytime-valid sequential hypothesis testing for distribution shift in streaming learning systems" by Khan and Syed. The manuscript's `note = {Version 2}` is the right disambiguation; adding the arXiv version suffix explicitly would help a reader who checks.

Two descriptive errors. First, E-SHIFT's Assumption 5.1(c) is a **sub-exponential** condition (Appendix A fits parameters $(\nu^2, c)$ with $\psi(\lambda) \le \nu^2\lambda^2/2$ for $|\lambda| < 1/c$), not sub-Gaussian; the manuscript understates the generality of the work it is criticising. Second, the manuscript says E-SHIFT "assumes unconditional exchangeability with calibration" — it assumes i.i.d. scores, which is stronger, and it *does* provide an α-mixing extension (its Theorem 6.2) for temporally correlated streams, which the manuscript does not mention while making autocorrelation the centrepiece of its own Layer 1.

The other two stated differences hold. E-SHIFT's own limitations section concedes "Proposition 6.1 provides only asymptotic validity for the bootstrap construction," so the "validity is asymptotic in the calibration size" charge is fair and is the authors' own. And the "wealth starts at frame one" charge is fair *within a segment*: E-SHIFT resets $M_t \leftarrow 1$ only after an alert (its Proposition 6.2), which is the same restart convention this manuscript applies to all baselines, so the criticism is about the absence of a mixture over onsets, not about a mischaracterised reset. I checked this specifically because §\ref{sec:ablation} leans on it, and it survives.

**Severity**: Minor
**Evidence Anchor**: text: "with a sub-Gaussian e-process built on a composite non-conformity score"
**Confidence**: 4 — read the full E-SHIFT v2 HTML including Appendix A and §8.1; I am confident on the sub-exponential point, less so on how much weight the α-mixing omission deserves.

### W6: The DKW figure attributed to Dvoretzky–Kiefer–Wolfowitz does not follow from the formula the paper states

The manuscript defines $\varepsilon_n(\delta) = \sqrt{\log(2/\delta)/(2n)}$ and then reports that at $n=2000$, $\delta=10^{-3}$ the DKW-inflated smallest attainable p-value is $0.051$. With those inputs the formula gives $\varepsilon = 0.04359$, so $1/2001 + \varepsilon = 0.0441$. I verified this against the repository's own `sentinel_e/conformal.py::dkw_inflation`, which implements the stated formula and returns 0.04359. The exact Beta figure is right — with $\delta_j \propto (j+1)^{-2}$ normalised by $\pi^2/6$, $q_0 = 1 - \delta_0^{1/2000} = 0.003695$, matching the quoted $0.0037$ to four places. So the consequence is that the improvement factor is roughly $12\times$, not the $14\times$ claimed in two places. The point being made survives comfortably; the number attributed to the cited inequality does not, and it recurs in the contributions list, in §\ref{sec:layer1} and implicitly in \AblDkwSpeedup. This overlaps D1 territory and I flag it here only because it is a numeric quantity attributed to a specific cited source.

**Severity**: Minor
**Evidence Anchor**: text: "the smallest usable p-value becomes $0.0037$ instead of the $0.051$ that Dvoretzky--Kiefer--Wolfowitz permits"
**Confidence**: 5 — arithmetic reproduced twice, once by hand and once against the paper's own implementation.

### W7: One bibliography DOI resolves to an unrelated paper

`shin2024edetectors` carries doi 10.51387/23-NEJSDS44, which resolves to Qiu and Wong, "Nature-inspired Metaheuristics for finding Optimal Designs for the Continuation-Ratio Models," NEJSDS. The E-detectors paper by Shin, Ramdas and Rinaldo is 10.51387/23-NEJSDS51, NEJSDS vol. 2 no. 2, pp. 229–260 (online December 2023). The volume, issue and page range in the entry are correct; only the DOI is wrong. No citation in this manuscript points to a non-existent work — I cross-checked every `\cite` key against `refs.bib` and found no dangling keys, and I verified the existence of every domain-side and sequential-analysis reference I could resolve. Five bib entries are present but never cited (`benjamini1995`, `kipf2017gcn`, `hamilton2017graphsage`, `hazan2007ons`, `vovk2003exchangeability`); the last of these is substantively relevant and its absence from the text is discussed under W8.

**Severity**: Minor
**Evidence Anchor**: text: "doi     = {10.51387/23-NEJSDS44},"
**Confidence**: 5 — both DOIs resolved through Crossref.

### W8: The conformal-test-martingale change-detection line is in the bibliography but absent from Related work

`vovk2003exchangeability` (Vovk, Nouretdinov and Gammerman, "Testing Exchangeability On-line," ICML 2003) sits in `refs.bib` uncited. That paper, and the line it began — inductive conformal martingales for change-point detection, conformal test martingales for retraining decisions, and the weighted variants for drift — is the closest existing combination of a conformal front end with a Ville-thresholded martingale back end, which is precisely the Layer 1 / Layer 2 architecture here. It does not occupy the paper's gap: I found no application of it to waste, dumping or pollution surveillance, and none of it carries the episode axis. But a domain reader who works in anytime-valid inference will expect it named and distinguished in §\ref{sec:related} rather than left in the bibliography, particularly given that the paper positions E-SHIFT rather than conformal test martingales as "closest to this work in spirit."

**Severity**: Minor
**Evidence Anchor**: absence: §\ref{sec:related}, "Conformal prediction" and "Anytime-valid inference and e-values" paragraphs — expected discussion of conformal test martingales for change-point detection (Vovk–Nouretdinov–Gammerman 2003 and successors); checked all \cite keys in sentinel_e.tex against refs.bib, the full Related work section, and the uncited-entry set
**Confidence**: 4 — the omission is verified mechanically; whether it rises above a citation gap is a judgement about what this venue's readers expect.

### S1: The Mivia-IWDD-500 description is accurate in every checkable particular

Every element of the dataset claim verifies. The corpus is Mivia-IWDD-500, balanced at 250 positive and 250 negative video sequences, with precise onset timestamp annotations on each positive video marking the moment dumping becomes visible, covering both static and dynamic disposal modalities under day and night conditions. Authorship is Greco, Ricciardi, Sansone and Vento; the venue is *Image and Vision Computing*, volume 174, article 106125, 2026 (doi:10.1016/j.imavis.2026.106125, DBLP journals/ivc/GrecoRSV26). The "first public dataset" characterisation is supported by the organisers' own survey, which records that all five prior video dumping corpora (Mahankali; Yun; Husni; Kim; Yu) "remain private assets." Adding the volume and DOI to the bib entry is the only change I would ask for. Given how often dataset descriptions in this area are transcribed from a webpage without checking, this is worth crediting.

**Evidence Anchor**: text: "with $250$ positive and $250$ negative surveillance videos and onset timestamps"

### S2: The anytime-valid gap claim survives adversarial checking

I treated this as the claim most likely to fail and searched accordingly: e-values and e-processes in environmental monitoring; conformal test martingales for pollution and water-quality detection; anytime-valid change detection in remote sensing, marine debris, oil spill and deforestation monitoring; and direct combinations of e-value/test-martingale/anytime-valid terminology with illegal dumping and waste surveillance. I found the machinery applied to clinical trials, A/B testing, streaming ML drift, industrial anomaly monitoring and conditional-independence testing, but no application to waste, dumping or pollution surveillance. The claim as written is, to the best of a reasonably thorough search, correct. One qualification the manuscript should absorb: *classical* sequential change detection has reached pollution surveillance — Guépié, Fillatre and Nikiforov applied their transient-change machinery to contamination detection in water distribution networks (IFAC 2012). So the accurate framing is that the *anytime-valid* branch has not reached this domain, not that sequential monitoring has not.

**Evidence Anchor**: text: "To our knowledge none of it has been applied to waste, dumping or pollution surveillance"

### S3: Attributions to cited works are quoted and characterised faithfully where I could check them

Two spot-checks that matter. The parenthetical quote attributed to the X3D entry is verbatim accurate: Almorsy and Torki report producing "only 6 false alarms on 30 negative videos (80% specificity)," and the manuscript's gloss — that this is specificity rather than a horizon rate — is the correct reading. The O(t) attribution to Saha and Ramdas is also exact: their paper states that "the exact all-start detector" costs O(t) per objective evaluation and O(t) memory at time t. I also confirmed that their change model is the standard persistent one, so the manuscript's characterisation of *that* work is fair even though its generalisation over the whole literature (W1) is not. The disclaimer paragraph crediting the PFA-controlled non-partitioned e-detector rather than claiming it is unusually scrupulous and should be kept.

**Evidence Anchor**: text: "a general mixture over active starts costs $O(t)$ per observation"

### S4: The dataset adapters are honest about provenance and refuse to fabricate

`sentinel_e/datasets/adapters.py` records, per corpus, the modality, access route, role in the evaluation and expected backbone output, and its stated design decision — that no adapter downloads or fabricates data, and that a missing intermediate raises `FileNotFoundError` carrying the exact recipe — is the right call for a paper whose empirical results are simulator-generated. The access descriptions are accurate (MIVIA and AerialWaste behind request forms, TACO's images on Flickr, MARIDA and MADOS on Zenodo, UAVVaste annotations on GitHub), and the Zenodo record IDs resolve. In a submission whose central limitation is the absence of real per-frame scores, an adapter layer that fails loudly rather than substituting synthetic data is the correct engineering answer and materially raises my confidence that the simulation boundary is not being blurred elsewhere.

**Evidence Anchor**: text: "the adapters raise an error rather than substituting synthetic data when those scores are absent"
---
## Author-side verification of the domain seat's checkable findings

- **W6 CONFIRMED, with the root cause identified.** At $n=2000$, $\delta=10^{-3}$:
  `dkw_inflation(2000, 1e-3) = 0.043592`, so the smallest usable p-value is
  $1/2001 + \varepsilon = 0.044091$, and the exact Beta level is
  `beta_calibration_levels(2000, 1e-3)[0] = 0.0036957`. The ratio is
  **11.93x, not 14x**. The stated $0.051$ is the $\delta = 10^{-4}$ value
  ($1/2001 + 0.049758 = 0.050258$): the two halves of the comparison were
  computed at different $\delta$. Occurs at sentinel_e.tex:229, 479 and 518.
- **W1 TRIANGULATED.** Independently raised by the Devil's Advocate (M2) and
  confirmed by our own literature check. Both seats' primary sources resolve.
  The domain seat adds two the DA did not name: Guepie/Fillatre/Nikiforov,
  "Detecting a suddenly arriving dynamic profile of finite duration", IEEE TIT
  2017; and Fuh & Mei, "Quickest change detection and Kullback-Leibler
  divergence for two-state hidden Markov models", IEEE TSP 63(18), 2015 --
  the last of which is a two-state HMM quickest-detection treatment and is the
  closest structural relative of our Equation 4 in the classical literature.
  The seat's scoping of the surviving claim ("new *within the anytime-valid
  e-process framework*") is the one we should adopt.
- **W8 CONFIRMED mechanically.** `vovk2003exchangeability` is in `refs.bib`
  and cited nowhere in the text; likewise `benjamini1995`, `kipf2017gcn`,
  `hamilton2017graphsage`, `hazan2007ons`.
- **W2/W3/W4/W5/W7** are external-source claims (contest team count, IVC issue
  date, ZeroWaste pagination, E-SHIFT's Assumption 5.1(c), the NEJSDS DOI).
  They are specific and checkable and the seat reports holding the primary
  records; each is re-verified individually during the revision, not accepted
  on the seat's word.
