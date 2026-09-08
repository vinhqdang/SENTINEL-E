# Simulated peer review

Artefacts from a five-seat simulated peer-review panel run over
`paper/sentinel_e.tex` under the ARS v3.6.2 reviewer sprint contract.

**What this is not.** These reviewers are language-model personas sharing one
model family and one provider. Role separation is a prompt-level construct; it
is not evidence of independent error processes, and this panel is
`NOT_CALIBRATED` — no measured false-negative or false-positive profile backs
its severity bands. Treat the output as a structured self-critique that finds
issues a careless read would miss, not as a substitute for journal review.

## Contents

| Path | What it is |
|---|---|
| `contract.json` | Frozen v2 sprint contract: six acceptance dimensions (D1--D6), their owners and priorities, and the mechanical failure-condition ladder F0--F5 that determines the editorial decision |
| `metadata.json` | Title, field and word count — the only paper information the Phase 1 seats were given |
| `phase1/*.md` | Each seat's paper-blind pre-commitment: what it will look for, and what will trigger warn / block / fatal on the dimensions it owns |

## Why Phase 1 exists

The load-bearing mechanism is that Phase 1 is called **before any seat sees the
manuscript**. Its purpose is to destroy the "read the paper, then rationalise a
standard that the paper happens to meet" drift path. In Phase 2 each seat is
handed back its own commitment as read-only data and must bind every warn or
block to a character-for-character substring of the trigger it wrote earlier —
so a seat cannot invent a lenient or a harsh standard after the fact.

Manuscript content is treated as untrusted data throughout: an imperative
sentence inside the paper is content under review, never an instruction to the
reviewer.
