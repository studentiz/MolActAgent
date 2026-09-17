# MolActAgent: Design and Implementation

> A complete, ground-up account of what MolActAgent is, why it is shaped
> the way it is, and how every component works in enough detail that a
> careful reader could re-implement the system from scratch.

This document is the senior authority for MolActAgent's architecture.
If a code change conflicts with what is written here, the document is
the senior authority unless we explicitly revise it first.

---

## Table of Contents

**Part I — Orientation**
1. [About this document](#1-about-this-document)
2. [What MolActAgent is](#2-what-molactagent-is)
3. [The thirty-second mental model](#3-the-thirty-second-mental-model)
4. [Reading paths through this document](#4-reading-paths-through-this-document)

**Part II — Architectural Philosophy**

5. [The two-actor system: witness and reasoner](#5-the-two-actor-system-witness-and-reasoner)
6. [What "similar" actually means](#6-what-similar-actually-means)
7. [Static value versus dynamic texture](#7-static-value-versus-dynamic-texture)
8. [The layered epistemic architecture](#8-the-layered-epistemic-architecture)
9. [Unification of classification and regression](#9-unification-of-classification-and-regression)
10. [Why this design earns its complexity](#10-why-this-design-earns-its-complexity)

**Part III — Implementation Walk-Through**

11. [Module map and data flow](#11-module-map-and-data-flow)
12. [Stage 1 — Data ingestion and validation](#12-stage-1--data-ingestion-and-validation)
13. [Stage 2 — Preprocessing](#13-stage-2--preprocessing)
14. [Stage 3 — Contextual similarity and neighbour selection](#14-stage-3--contextual-similarity-and-neighbour-selection)
15. [Stage 4 — Adaptive top-M class selection](#15-stage-4--adaptive-top-m-class-selection)
16. [Stage 5 — Per-tuple classification](#16-stage-5--per-tuple-classification)
17. [Stage 6 — Validation, override policy, retry](#17-stage-6--validation-override-policy-retry)
18. [Stage 7 — Aggregation, calibration, attribution](#18-stage-7--aggregation-calibration-attribution)
19. [The CCR regression backend](#19-the-ccr-regression-backend)
20. [The LLM client](#20-the-llm-client)
21. [Pipeline orchestration](#21-pipeline-orchestration)
22. [Report rendering](#22-report-rendering)
22.5. [The v0.3 layered API surface (estimators / metrics / tools / pp / datasets)](#225-the-v03-layered-api-surface-estimators--metrics--tools--pp--datasets)

**Part IV — Reproducibility and Configuration**

23. [Hyperparameter reference](#23-hyperparameter-reference)
24. [LLM configuration](#24-llm-configuration)
25. [Building the system from scratch](#25-building-the-system-from-scratch)
26. [What good behaviour looks like](#26-what-good-behaviour-looks-like)

**Part V — Empirical Foundations**

27. [Benchmarks](#27-benchmarks)
28. [Known limitations and open questions](#28-known-limitations-and-open-questions)
29. [Status and extensibility guidance](#29-status-and-extensibility-guidance)

---

# Part I — Orientation

## 1. About this document

The original `docs/design.md` was a philosophy paper: it explained
the architectural commitments and the reasoning that produced them
but said almost nothing about how the running system actually
implements them. This rewrite keeps the philosophy — the reasoning
narrative is genuinely useful for anyone extending the system — and
adds the missing implementation detail in a separate, longer Part
III. The two stand on their own:

- **Read Part II** when you want to understand *why* the system is
  shaped the way it is.
- **Read Part III** when you want to understand *how* a specific
  component works, *what* its inputs and outputs are, and *which
  formulas* govern its behaviour.
- **Read Part IV** when you want to reproduce the system from
  scratch or pin down a specific hyperparameter.

This document is exhaustive on purpose. MolActAgent is small enough
that it ought to be possible to keep the entire design in one
person's head, and this is the document that lets that happen.

Where the text says "the design rejects X in favour of Y", X was
almost always a serious early proposal worth understanding — not a
strawman invented to make Y look good. Tracking *why* X failed is
what sharpens the sense of *why* Y is right.

## 2. What MolActAgent is

MolActAgent is a small, transparent prediction toolkit for tabular
data. You hand it a labelled **reference dataset** and one or more
**unlabelled test rows**; it returns a verdict (a class label or a
continuous value), a calibrated confidence score, and a
human-readable Markdown / HTML report explaining *why* — every
claim grounded in real numbers from your reference set.

Two task families are supported through a single unified pipeline:

- **Classification**, for any number of distinct classes.
  MolActAgent adaptively narrows the comparison to the M classes in
  contention for each test sample (typically M=2; sometimes M=3
  near a boundary) and runs a contrastive 1-test-vs-M-references
  analysis per LLM call.
- **Regression**, for continuous targets. MolActAgent reformulates
  regression as *two* multi-class classifications — one over
  equal-frequency target bins, one over unsupervised feature-space
  clusters — then aggregates the picks of both views into a
  confidence-weighted numeric prediction. This backend is the
  Cross-Classification Regressor (CCR).

MolActAgent is *not* a high-throughput model. It is intended for
the small-data regime where each prediction can pay for several
LLM calls in exchange for an explanation a domain expert can read
and check. The LLM is the reasoner; Python is the witness that
keeps the LLM honest.

## 3. The thirty-second mental model

```
   Reference data (N labelled rows)        Test sample (1 unlabelled row)
                │                                       │
                ▼                                       ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  1. WITNESS LAYER (Python)                                  │
   │     - Standardise features.                                 │
   │     - Compute contextual similarity = blend of feature      │
   │       distance and neighbourhood-of-neighbourhoods overlap. │
   │     - For each candidate class, take the k contextually-    │
   │       closest references.                                   │
   │     - Pick the M classes "in contention" (default M=2).     │
   │     - For each (M+1)-tuple of (test, ref_1, …, ref_M):      │
   │         tally per-feature alignment counts                  │
   │         pick a deterministic baseline class                 │
   │         pull each reference's k_local neighbourhood for     │
   │         "texture" (cluster interior vs boundary)            │
   └──────────────────────────────────┬──────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  2. PROMPT ASSEMBLER (Python)                               │
   │     Lays out for each tuple: discriminative-feature table,  │
   │     deterministic alignment tally as ground truth,          │
   │     per-reference neighbourhood blocks, JSON output schema. │
   └──────────────────────────────────┬──────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  3. REASONER (LLM)                                          │
   │     One call per tuple → JSON: label, confidence,           │
   │     3-6 named drivers with per-driver justification,        │
   │     supporting/opposing features, summary, counter-evidence │
   └──────────────────────────────────┬──────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  4. AGGREGATOR (Python)                                     │
   │     Tally weighted votes across tuples → final label,       │
   │     calibrated confidence, weighted driver attribution.     │
   │     One optional EXTRA LLM call writes the Markdown report. │
   └─────────────────────────────────────────────────────────────┘
```

For regression, replace the per-tuple inner loop with two parallel
classification passes (target bins and feature clusters), then
combine their picks via a confidence-weighted mean of the actual
target values. The witness, prompt assembler, reasoner, and
aggregator stages are *all the same*; only the class system and
the post-processing of picked references differ.

## 4. Reading paths through this document

Pick the path that matches your goal.

| Goal | Read |
|---|---|
| "I want to understand the architectural philosophy." | Part II in order. Optionally skim §27. |
| "I want to extend a feature." | §11 (module map), then the relevant §12-22 stage, then §28-29 (constraints). |
| "I want to reproduce MolActAgent in a different language." | §11 first, then Part III start to finish, then Part IV as the implementation reference. |
| "I want to debug a specific output." | §22 (report layout) → §18 (aggregation) → §16 (per-tuple) — work outward from the symptom. |
| "I want to tune MolActAgent for my dataset." | §23 (hyperparameter reference) and §27 (benchmarks). |

The document is dense — that is deliberate. Most sections include
worked examples and explicit code references so you can find the
canonical implementation immediately if the prose is ambiguous.

---

# Part II — Architectural Philosophy

This part is a near-verbatim continuation of the original design
notes, reorganised for newcomer readability and tied to specific
implementation hooks where that helps. The philosophy is
load-bearing: every detail in Part III either implements one of
these commitments or reflects a trade-off forced by them.

## 5. The two-actor system: witness and reasoner

MolActAgent is a two-actor system, by intentional design:

- **Python is the witness.** It computes things that are too simple
  to get wrong and too embarrassing to outsource: distances,
  neighbour sets, alignment counts, vote tallies. These are
  presented to the LLM as ground truth.
- **The LLM is the reasoner.** It looks at the witness's report
  alongside the raw evidence and produces a verdict with a
  rationale. It is allowed to disagree with the witness only when
  it can clearly justify the disagreement; otherwise the witness
  wins.

### 5.1 How we arrived at this division

The natural temptation, when building a system that combines code
and an LLM, is to push as much "intelligence" into the code as
possible and use the LLM only as a fallback. That instinct comes
from years of treating LLMs as unreliable narrators — better to
nail down what we can in deterministic logic and let the LLM only
handle the residual.

We tried this in early multi-class design discussions. Concrete
proposals included: an epsilon for tie-breaking when the alignment
between Test and two references was nearly equal; margin gates to
reject "weak overrides" by the LLM; soft alignment with continuous
weights instead of a discrete tally. Each addition was individually
sensible. Together they accumulated into something that pretended
to be more rigorous than the underlying system actually was.

The realisation that broke this pattern: the LLM is going to
reason fuzzily about everything we hand it anyway. Adding numerical
precision to one stage of a deliberately fuzzy pipeline does not
make the pipeline more accurate — it makes it dishonest about its
own nature, and it shifts judgement calls into Python where they
become invisible to the only component capable of actually
exercising judgement.

So the rule we settled on is the negation of the original instinct:
**Python should compute only what it would be embarrassing to get
wrong; everything else belongs to the LLM.** Distances are
embarrassing to get wrong. "Is this driver strong enough to
override the baseline?" is not — it is a judgement call, and
judgement calls go to the reasoner.

### 5.2 What this means in practice

A test for any new feature: which actor does it belong to?

- If it is a number that needs to be right, it is a Python concern.
- If it is a judgement call, it is an LLM concern.
- If it is something in between — a heuristic that is *usually*
  right — it probably belongs in Python as a *witness statement*
  (a heuristic value Python computes and presents to the LLM as
  ground truth) with the LLM positioned to override it when
  warranted.

If you find yourself reaching for a hand-tuned numerical heuristic
to make the agent decide something borderline, ask first whether
you should instead surface the borderline situation honestly to the
LLM and let it reason about it. The LLM is the only component in
the system that is allowed to be fuzzy.

### 5.3 Where this shows up in the code

- The deterministic alignment tally is computed in
  `_classify_tuple` and surfaced as ground truth in the per-tuple
  prompt — see [`molactagent/agent.py`](../molactagent/agent.py)
  lines around `n_aligned`.
- The "soft baseline check" in `_check_tuple` flags an LLM verdict
  that contradicts the alignment-tally baseline AND lacks
  decisively-strong opposing drivers, then asks the LLM for one
  revision; the final clamp lives in `_finalize_tuple`.
- Per-tuple `class_scores` come from the alignment tally, **not**
  from the LLM's stated `strength` numbers — so per-tuple scoring
  remains reproducible. The LLM's drivers power explanations and
  attribution; they do not override Python's tally.

## 6. What "similar" actually means

The standard view treats similarity as point-wise: A and B are
similar to the extent that their feature vectors are close in some
metric. Under this view, a sample is just a point in feature space.

MolActAgent rejects this as incomplete. A sample is not a point —
it is a point plus the local context that surrounds it. Two
samples are similar to the extent that their *expanded identities*
(point plus neighbourhood structure) line up.

### 6.1 How we arrived at this

We started where most ML systems start: similarity is feature-space
distance, full stop. This is the formulation underlying every
KNN-family method.

The thing that destabilised it was a worked example we kept coming
back to. Suppose A is at distance 0.6 from B in feature space —
fairly dissimilar by the standard view. But:

- A's four nearest neighbours are C, D, E, F at distances 0.5, 0.5,
  0.1, 0.0.
- B's four nearest neighbours are C, H, I, J at distances 0.5, 0.5,
  0.1, 0.0.
- The cross-distances D-H, E-I, F-J are all ≤ 0.01.

A and B's neighbourhoods are nearly identical even though A and B
themselves are 0.6 apart. By every meaningful test except the raw
A-to-B distance, they sit in the same region of feature space and
should behave similarly with respect to whatever function we are
trying to predict. Pure feature distance misses this entirely.

This matches how human experts actually reason about similarity. A
chemist saying "compound X is similar to compound Y" does not mean
"X's molecular weight is close to Y's." They mean "X belongs to
the same structural family as Y; both sit among similar molecules
behaving in similar ways." Similarity is *being-in-a-similar-place*,
not *having-similar-numbers*.

### 6.2 What we considered and rejected

The first instinct was to operationalise this immediately: write
down a bipartite matching cost between A's k-neighbour set and B's
k-neighbour set, blend it with feature distance at some weight,
and move on. Variants in the same breath were Earth Mover's
Distance, average minimum cross-coverage, and a Hungarian-algorithm
minimum assignment cost.

This was the wrong move, even though the formulas are reasonable.
It operationalised before we had internalised what was being
measured. The risk: we get attached to a specific computation and
start defending it (or its weights) instead of staying open to the
simpler explanation that the philosophy itself requires.

The simpler explanation, once we let it surface: a sample's
identity is partly constituted by its surroundings. A and B are
similar because their *expanded identities* line up — and asking
which bipartite-matching score is the right way to compute that is
downstream of accepting the philosophical move first.

### 6.3 What the design commits to

A blend of two quantities:

- **Feature similarity** — a conventional inverse-distance score
  between A's and B's feature vectors.
- **Neighbourhood similarity** — a measure of how isomorphic A's
  k-nearest-neighbour set is to B's.

Both are necessary and neither dominates. The blend is symmetric:
`sim(A, B) = sim(B, A)`. The default mixing weight is 50/50,
adjustable per dataset, but the weight is a tuning parameter, not a
load-bearing piece of the design.

The exact form of "neighbourhood similarity" — bipartite matching,
mean cross-coverage, something else — is an implementation choice.
Different choices have different computational profiles and
slightly different sensitivities, but they all express the same
underlying claim: *we measure how well one neighbourhood
substitutes for the other*. We chose the simplest formula that
captures this; §14 documents it precisely.

### 6.4 Where this shows up in the code

- Feature similarity, neighbourhood similarity, and the contextual
  blend are all in
  [`molactagent/neighbors.py`](../molactagent/neighbors.py): see
  `NeighborFinder.fit`, `neighbourhood_similarity`,
  `contextual_similarity_to_test`, and `find_top_k_references`.
- The 50/50 default lives at `DEFAULT_SIMILARITY_BLEND = 0.5`; the
  per-reference local k_local lives at `DEFAULT_K_LOCAL = 3`.

## 7. Static value versus dynamic texture

A reference's contribution to the LLM's reasoning has two
components, and they are conceptually distinct:

- **The static value.** What the reference IS — its class label,
  or its target value. This captures the reference's own nature.
- **The dynamic texture.** How values change in the reference's
  neighbourhood — the local trend in target values, or the local
  composition of class labels. This captures the reference's
  surroundings, not the reference itself.

Both are useful; both are different kinds of evidence. The static
value answers "what does this reference exemplify?". The dynamic
texture answers "how does that exemplification generalise to
nearby positions?".

### 7.1 How we arrived at the distinction

Until we tried to extend the framework to regression, we had been
treating each reference as carrying just one piece of information:
its label (for classification) or its target value (for
regression). This is the implicit model in standard KNN.
Regression-by-KNN means "average the target values of the k
nearest neighbours." It does not ask, and cannot ask, *whether the
function being predicted is varying or stable around each
neighbour*.

The shift came from a worked example. Imagine R1 has target value
10, with neighbours whose values trend higher in some feature
direction. R2 has target value 14, with neighbours whose values
trend lower. Test sits structurally near both. Pure KNN predicts
(10 + 14) / 2 ≈ 12. But if R1's gradient says "values rise toward
Test" and R2's gradient says "values fall toward Test," the test
value might well be 16 — *above* both references — because the
test sits at a local peak.

KNN cannot reach this conclusion because it is averaging static
values. The information needed to reach it lives not in R1's value
or R2's value, but in *how values vary around each*. That
information is qualitatively different: it is dynamic where they
are static, directional where they are positional.

Once that distinction was named, several apparent paradoxes
resolved. Two references can have the same static value but
completely different dynamic texture (one on a rising slope, one
on a falling slope). They contribute the same evidence about
"what the function looks like nearby" only if you ignore texture;
if you include texture, they contribute very different evidence.

### 7.2 The over-engineering trap we narrowly avoided

Once the distinction was clear, the temptation was to build
texture into the system as a *first-class object*: a "gradient
computer" or "purity scorer" that Python would calculate and pass
to the LLM alongside the static values. We started designing what
its schema should look like, what units it should report in, how
it should handle non-monotonic neighbourhoods, and so on.

This was the same trap as §6. The relevant question is not "how
do we compute texture?" but "what does the LLM need to perceive
texture?". And the answer, once asked clearly, is: *the underlying
data*. If you show the LLM a reference's features, its value, AND
the features and values of its k_local nearest neighbours, the LLM
perceives the texture by reading. Stable cluster interior, sharp
boundary, monotone gradient, local peak — these are patterns the
LLM picks up from the assembled data without any explicit
summarisation.

This is consistent with the core posture from §5. Texture is not
a number that would be embarrassing to get wrong; it is a
qualitative pattern. Qualitative patterns belong to the reasoner.
Python's job is to assemble the right context; the LLM's job is to
perceive its shape.

### 7.3 The asymmetry of texture is a feature

Feature similarity and neighbourhood similarity (§6) are symmetric
between two samples — the relation is mutual. Texture, in
contrast, is asymmetric: A's texture and B's texture are different
objects, each tied to its own sample, and they need not agree,
mirror, or even point in compatible directions.

The reconciliation: we are not supposed to combine asymmetric
evidence by averaging — that would be classical KNN's mistake. We
are supposed to *synthesise* it through reasoning. Each reference
brings its own perspective on the local landscape. Multiple
perspectives are richer than one because they triangulate.

For classification, the analogue of gradient is *neighbourhood
composition*. R1 might be class A surrounded entirely by other
A's ("deep in A territory"), while R2 might be class A but with
several class B neighbours ("on the A/B boundary"). The LLM should
weight R2 less heavily as evidence for "Test is class A" than R1,
even though both are formally class-A references. This is also
asymmetric, also directional in spirit, and also implicit in
showing the LLM each reference's neighbours rather than computing
a separate "boundary-ness" score.

### 7.4 Where this shows up in the code

- The texture blocks for each chosen reference are assembled in
  `_classify_tuple` and passed via `reference_neighbourhoods=`
  into `PromptBuilder.build_tuple_prompt`. They become the
  "PER-REFERENCE LOCAL CONTEXT" section of the prompt.
- For regression, direction (`increases` / `decreases` / `mixed`)
  is computed from the per-feature Pearson correlation with the
  target across the training set — see
  `_synthesize_driver_attribution` in
  [`molactagent/cross_classification.py`](../molactagent/cross_classification.py).
  This is a witness statement: a number Python computes and hands
  to the report renderer; the LLM is free to disagree.

## 8. The layered epistemic architecture

The system is composed of three layers with deliberately different
epistemic commitments. Understanding these layers is the key to
not over-engineering — most over-engineering proposals turn out to
be attempts to push commitments into a layer that should not have
them.

1. **Filter (Python).** Uses contextual similarity to prune the
   reference set down to a small number of candidates per test
   sample. This layer *assumes local smoothness* — it presumes
   that contextually similar samples are informative about each
   other. The assumption is a heuristic for narrowing the search
   space and eliminating noise. It is allowed to be a little
   wrong.

2. **Evidence assembler (Python).** Builds the prompt by laying
   out the test sample's features alongside each filtered
   reference's features, value, and neighbourhood. This layer is
   *agnostic* — it makes no assumptions about the function being
   predicted; it just presents what the data shows.

3. **Reasoner (LLM).** Reads the assembled context and produces
   a verdict. This layer makes *no smoothness assumption at all*.
   The LLM is free to notice "the gradients in this neighbourhood
   disagree, so something non-smooth is happening here" or "this
   reference looks similar but its neighbourhood is full of the
   other class — treat it sceptically."

### 8.1 Why this matters

Classical KNN regression *commits* to local smoothness — it
averages the y-values of neighbours and stops there. There is no
layer above the averaging that can perceive non-smoothness, so KNN
regression silently smooths over discontinuities. MolActAgent uses
local smoothness only to *select what to look at*, then asks an
actor capable of perceiving non-smoothness to do the actual
prediction. Those are different epistemological positions.

The layered view also reframes what counts as a "weakness" in the
system. The fact that the filter has a smoothness assumption is
not a weakness if the layer above can correct for the cases where
it is wrong. Failures cascade *upward* to be flagged, not
*downward* to be silently propagated. A small filter mistake (a
barely-relevant reference passes through) becomes the LLM's chance
to discount that reference; a severe filter failure (no good
candidates at all) becomes the LLM's chance to express low
confidence.

### 8.2 A useful test for new features

Which layer does a new feature belong to?

- If it computes a number that needs to be right: filter or
  assembler.
- If it makes a judgement call: reasoner.
- If it is something in between: it probably belongs in the
  filter *as a heuristic*, with the assembler honestly surfacing
  it and the reasoner positioned to override.

The most common failure mode in proposals so far has been to put
something in Python that should have been left to the LLM,
*because the proposal felt safer when "computed" rather than
"reasoned about."* That instinct is wrong here.

## 9. Unification of classification and regression

Once the layered view is in place, classification and regression
collapse into a single mechanism:

> Find references whose context resembles the test sample's. For
> each reference, show the LLM not just the reference itself but
> the reference's surroundings. Ask the LLM to locate the test
> sample within the assembled landscape.

That description does not mention "label" or "value." The LLM's
cognitive task is the same in both cases: read the assembled
landscape, locate the test sample, report what it finds at that
location. The format of the report (a class name vs a number) is
set by what kind of value lives at each reference, not by anything
the LLM does differently.

### 9.1 The proposals we rejected

Three rejected proposals are worth knowing about because they
recur:

**Anchored / bounded / free output constraints.** An early proposal
asked whether the LLM should be allowed to predict any number, or
only numbers within the observed reference range, or only numbers
chosen from a finite set of "anchors". This was wrong because the
constraints were trying to substitute for the LLM's reasoning. If
the LLM is the decision-maker, constraining its output to a
discrete set assumes the LLM cannot pick a sensible number on its
own — which inverts the design philosophy.

**An "Inconclusive" sentinel for regression.** Classification has
a clean categorical "I don't know" output (`Inconclusive`). The
analogue for regression was proposed to be `None` plus a flag, or
a wide interval, or an error. This was wrong because regression
does not need a categorical sentinel: uncertainty is natively
expressible in a regression output ("around 13, give or take 2").
The fuzziness lives in the texture of the answer itself, not in a
sentinel value. (CCR retains a `predicted_value=None` refusal as a
true *I cannot answer this* — see §19.6 — but that is a *failure
mode*, not the analogue of `Inconclusive`.)

**A "structurally similar but numerically far" tension.** An early
worry was that the contextual-similarity layer would select
references the LLM could not defend in its reasoning. The proposal
was to add a `selection_rationale` field, or to show the LLM the
structural-similarity evidence directly. This was wrong because
the worry itself was misframed: the 50/50 blend means structural
similarity *supplements* feature similarity, not *replaces* it.
References that pass the filter are still reasonably close
numerically.

### 9.2 What stays specialised

There is one specialisation that classification keeps and
regression drops:

- **Classification** uses an adaptive top-M step to narrow the
  candidate classes for each test sample (typically M=2;
  sometimes M=3) before forming the (M+1)-tuple. This keeps the
  prompt small when the test sample clearly belongs to a known
  class.
- **Regression**, in CCR, runs *two* classifications under the
  hood (target-bin and feature-cluster). The flat top-k pool is
  used internally to pick references after each classification
  picks a winning class.

Apart from that, the pipelines coincide. Every other component —
similarity computation, neighbourhood assembly, prompt structure,
reasoning step, aggregation — is shared.

### 9.3 The graceful in-between case

A bonus of the unification: ordinal classification (labels with a
natural order, like `low / medium / high`) falls out without
special handling. The LLM perceives label progression as a texture
much like a gradient, and reasons about the test sample's position
within an ordered series. That is not a special case to design
around; it just emerges from the same "read the context, perceive
the pattern" mechanism.

## 10. Why this design earns its complexity

The design is more elaborate than vanilla KNN, the prompts are
larger, and we trust the LLM to do qualitative landscape-reading.
That investment buys three things classical methods cannot deliver:

1. **Robust to false friends.** A sample that is numerically close
   to the test sample but lives in a different functional regime
   gets discounted by the neighbourhood-similarity component.
   Pure feature distance would miss this. False friends are a real
   problem in small-data domains where feature space is sparse
   and coincidental closeness is common.

2. **Reads landscapes the LLM can verify.** Because the LLM sees
   each chosen reference's surrounding data, it can perceive
   gradients, boundaries, cliffs, and stable plateaus directly
   from the prompt. Pure KNN sees only neighbour values, not
   surrounding texture, and so cannot distinguish a sample at a
   peak from a sample on a slope.

3. **Honest about non-smoothness.** Classical KNN regression
   silently smooths over discontinuities by averaging across them.
   The LLM, seeing the references and their neighbourhoods, can
   notice that the function is jumping and reflect that in its
   verdict and confidence — neither rigidly smooth nor recklessly
   coarse.

Whether the trust pays off is an empirical question — see §27.

---

# Part III — Implementation Walk-Through

This part documents how the system actually runs, end to end. Each
section corresponds to a stage of the pipeline; the formulas,
control flow, and edge-case handling are spelled out fully so
re-implementers do not have to read the source to recover them.

## 11. Module map and data flow

MolActAgent is organised into **two physical layers** that match the
"engine vs. user-facing API" split in the design philosophy. Read
this section first if you've never opened the source tree before —
the file layout looks bigger than it is, and a quick mental map
makes the rest of Part III much faster to navigate.

### 11.1 Engine — flat root-level modules

The engine is twelve Python files at the package root. These are
the "Python is the witness, LLM is the reasoner" implementation
from §5; everything else in the package builds on them.

| File | Role |
|---|---|
| [`molactagent/agent.py`](../molactagent/agent.py) | The `MolActAgent` class — the user-facing predictor for both classification and regression (regression delegates to CCR). |
| [`molactagent/cross_classification.py`](../molactagent/cross_classification.py) | The `CrossClassificationRegressor` (CCR) — the regression backend. |
| [`molactagent/neighbors.py`](../molactagent/neighbors.py) | The `NeighborFinder` — contextual similarity, per-class k-NN, adaptive top-M, flat top-k. |
| [`molactagent/preprocessing.py`](../molactagent/preprocessing.py) | `DataPreprocessor` — scaling and missing-value imputation. |
| [`molactagent/prompts.py`](../molactagent/prompts.py) | `PromptBuilder` — every prompt the system sends to the LLM. |
| [`molactagent/llm.py`](../molactagent/llm.py) | `LLMClient` — OpenAI-compatible wrapper with retries, throttling detection, and statistics. |
| [`molactagent/io.py`](../molactagent/io.py) | Data loading and validation: CSV/TSV/Excel/Parquet/JSON readers, `validate_inputs`, `prepare_for_agent`. |
| [`molactagent/config.py`](../molactagent/config.py) | LLM configuration: file/env/keyword resolution, JSONC parsing, profile selection, `LLMConfig`, `build_llm_client`, starter file generation. |
| [`molactagent/pipeline.py`](../molactagent/pipeline.py) | The high-level `run_prediction` (and `run_classification` / `run_regression` aliases) orchestrating the whole flow. |
| [`molactagent/report.py`](../molactagent/report.py) | Markdown / HTML report rendering. |
| [`molactagent/cli.py`](../molactagent/cli.py) | The `molactagent` command-line interface. |
| [`molactagent/utils.py`](../molactagent/utils.py) | Inner array-level validator used by `MolActAgent.fit`. |

### 11.2 Public API surface — themed sub-packages

On top of the engine sits the v0.3 layered Python API. These are
**thin** projection / orchestration layers — they own no
prediction logic of their own (every numeric output ultimately
comes from the engine above). The split is by *what the user is
trying to do*, not by what the engine internally does.

| Sub-package | Role |
|---|---|
| [`molactagent/result.py`](../molactagent/result.py) | `ClassificationResult` / `RegressionResult` / `DriverEntry` — typed wrappers around the engine's per-sample dicts. `to_dict()` round-trips to the engine's exact wire format. |
| [`molactagent/estimators/`](../molactagent/estimators/) | sklearn-style estimators (`MolActAgentClassifier`, `MolActAgentRegressor`) with `fit` / `predict` / `predict_proba` / `predict_with_confidence` / `score`. |
| [`molactagent/metrics/`](../molactagent/metrics/) | Classification + regression metric suites (`classification_metrics`, `regression_metrics`, `format_comparison_table`). Typed `Report` dataclasses with `to_table()` / `to_dict()` / `to_dataframe()` helpers. |
| [`molactagent/tl/`](../molactagent/tl/) | Downstream "tools" — Scanpy-style. `loocv`, `kfold_cv`, `compare_baselines`, `confidence_calibration`, `aggregate_drivers`. |
| [`molactagent/pp/`](../molactagent/pp/) | Preprocessing — Scanpy-style. Chemistry (`smiles_to_features`, `descriptors`, `fingerprints`, `scaffold`) plus a tabular `load_dataset` convenience. |
| [`molactagent/datasets/`](../molactagent/datasets/) | Bundled-example accessors: `load_iris_flowers`, `load_penguins`, `load_bslt_regression`. |

### 11.3 How the layers connect

```
                 USER CODE
              ┌──────┴──────┐
              │             │
          High-level     CLI
          (run_*)        (molactagent ...)
              │             │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │             │       Public API surface (thin)
          Estimators    Pipeline
        (Class/Reg)     orchestration
              │             │
              └──────┬──────┘
                     ▼
              ┌────────────┐
              │  ENGINE    │       Witness / Reasoner / Aggregator
              │  agent.py  │       (the philosophy in §5–§9)
              └─────┬──────┘
                    │
        ┌───────────┼────────────┬───────────┬────────────┐
        ▼           ▼            ▼           ▼            ▼
   neighbors.py  prompts.py   llm.py    cross_cls.py   report.py
   (witness)    (assembler)   (LLM)     (CCR for       (output)
                                         regression)
```

Three orthogonal concerns route through the engine:

- `ma.pp.*` (preprocessing) sits **before** the engine — converts
  user data into the `(X, y, feature_names, descriptions,
  background)` shape the engine expects.
- `ma.tl.*` (tools), `ma.metrics`, and the result wrappers sit
  **after** the engine — orchestrate engine calls, project the
  outputs into convenient shapes, and compute metrics.
- `ma.datasets.*` is **adjacent** to the engine — provides
  ready-made `(X_train, X_test, y_train, ...)` bundles drawn from
  the bundled examples.

### 11.4 The full call graph for a `molactagent run` invocation

A typical `molactagent run` call traces through the modules like
this (this is the engine flow — the `ma.tl` / `ma.metrics` /
`ma.pp` paths are similar but differ in their entry point):

```
   CLI                    molactagent.cli.main
                                 │
   Pipeline orchestration        ▼
                          molactagent.pipeline.run_prediction
                                 │
                ┌────────────────┼─────────────────────────┐
                ▼                ▼                         ▼
          io.load_table   io.validate_inputs        config.build_llm_client
                                 │                         │
                                 ▼                         ▼
                       io.prepare_for_agent          llm.LLMClient
                                 │
                                 ▼
                          agent.MolActAgent
                              .fit(X, y)
                                 │
                       ┌─────────┴──────────┐
                       │                    │
                  classification        regression
                       │                    │
                       ▼                    ▼
              for each test sample   CCR.fit(X, y) at
              .predict_one(x)        agent fit-time;
                       │             then per test sample:
                       ▼             two MolActAgent passes
        neighbors.NeighborFinder     + aggregation.
        .find_local_references
        .select_top_m_classes
                       │
                       ▼
        For each (M+1)-tuple:
          - alignment tally
          - texture blocks (k_local nbrs per ref)
          - prompts.build_tuple_prompt
          - llm.chat (parallel via ThreadPoolExecutor)
          - JSON parse + _check_tuple validation
          - up to N retries with feedback
                       │
                       ▼
                aggregate_results
                       │
                       ▼
        (Optional) generate_explanation
        — one extra LLM call per sample
                       │
                       ▼
              report.render_*
                       │
                       ▼
        summary.csv, interpretability.{md,html},
        per-sample HTML/Markdown/JSON
```

The remainder of Part III walks each stage in detail.

## 12. Stage 1 — Data ingestion and validation

### 12.1 Loading

Inputs may be DataFrames or paths to one of the supported tabular
formats:

```
.csv   .tsv   .txt   .xlsx   .xls   .parquet   .pq   .json   .jsonl
```

Plus `-` to read CSV (or `--stdin-format=tsv`) from standard
input. The reader is `io.load_table`. Optional artefacts:

- `feature_descriptions` — JSON or YAML mapping every column
  (including the label column) to a short string. Loaded by
  `io.load_feature_descriptions`.
- `background` — free-text domain context. Loaded by
  `io.load_background`. A leading `@` means "treat the rest as a
  file path" — this is the convention the CLI exposes via
  `--background @notes.txt`.

### 12.2 The `validate_inputs` contract

`io.validate_inputs(reference_df, test_df, label_column, *, ...)`
collects *every* problem before returning, so the user sees a
complete list rather than fixing one error at a time. It returns
a `ValidationResult` with three fields:

- `is_valid: bool` — `True` only when `errors` is empty.
  Warnings do not block.
- `errors: list[str]` — blocking problems, each phrased as a
  complete sentence saying *what is wrong* AND *what to do about
  it*.
- `warnings: list[str]` — non-blocking diagnostics.
- `summary: dict[str, Any]` — useful facts gathered along the way
  (`n_samples`, `class_labels`, `task`, `target_range`,
  `n_features`, etc.) that downstream code can reuse without
  recomputing.

The validator never raises on data semantics; it only raises
`TypeError` / `ValueError` for blatant API misuse (wrong argument
type, etc.).

#### Validation checks (classification)

- Label column exists in the reference; ≥ 2 distinct classes;
  no missing labels; ≥ 1 sample per class.
- A warning when there are > 10 distinct classes (the LLM gets
  shaky with too many candidates per call).
- For each class, a warning when its sample count is below `k`.
- Feature columns: present in both reference and test;
  test-extras get a warning; numeric coercion attempted; columns
  that are >50% non-numeric become errors; all-NaN and
  zero-variance columns become warnings.
- `feature_descriptions` (if provided) must cover every column,
  including the label column; extras are warnings.
- `id_column` (if provided) must exist in both files, be
  non-NaN, be unique within each file, AND not overlap between
  reference and test.

#### Validation checks (regression)

Auto-detected when the label column is numeric AND either has
non-integer values OR has more than 10 distinct values.

- ≥ 3 distinct numeric target values (otherwise the user is
  almost certainly facing a classification task instead).
- Non-degenerate range (`max > min`).
- ≥ 2 reference rows.
- Same feature-column checks as classification.

### 12.3 `prepare_for_agent`

`io.prepare_for_agent` runs *after* validation passes and converts
the validated DataFrames into numpy arrays plus metadata,
returning a `PreparedData` dataclass with:

- `X_ref`, `X_test` — `float64` numpy arrays of features in the
  reference column order.
- `y_ref` — class labels as strings (classification) or floats
  (regression).
- `feature_names` — list of column names in `X_ref` order.
- `class_labels` — sorted, canonical class order for
  classification; empty list for regression.
- `target_range` — `(min, max)` for regression; `None` for
  classification.
- `test_index` — the test DataFrame's index (after
  `id_column`, if any, has been promoted to the index).
- `test_true_labels` — captured for display only when the test
  CSV happens to include the label column; never used during
  prediction.

`prepare_for_agent` drops columns that are entirely NaN in the
reference set, coerces features to numeric (NaN where coercion
fails — handled later by the preprocessor's median imputation),
and stringifies classification labels.

## 13. Stage 2 — Preprocessing

[`molactagent/preprocessing.py`](../molactagent/preprocessing.py)
houses `DataPreprocessor`, a thin wrapper around scikit-learn's
scalers with one extra concern: **median imputation** for missing
values.

### 13.1 The fit/transform contract

```
fit_transform(X_ref):
    if any NaN in X_ref:
        for each column with NaN:
            replace NaNs with that column's median (computed over the
            non-NaN entries of that column only)
    if scaler is "standard":   apply StandardScaler.fit_transform
    elif scaler is "minmax":   apply MinMaxScaler.fit_transform
    elif scaler is "none":     return a copy of the imputed array
    return the resulting array

transform(X_test):
    for each column in X_test that contains NaNs:
        if we observed that column had NaNs at fit time, OR we did
        not but happen to encounter NaNs now, fill them with the
        column median computed at fit time.
    apply the fitted scaler (or return a copy if scaler="none").
```

### 13.2 Scaling rationale

Scaling is on by default (`StandardScaler`) because the project's
contextual-similarity blend (§14) uses Euclidean distance over
standardised features. Without scaling, a single high-magnitude
column would dominate the distance and silently destroy the
structural-similarity signal.

Two callers use scaled data slightly differently:

- The neighbour search and the `_X_ref_scaled` array used for
  contextual similarity always run on standardised features.
- The values shown in the LLM prompt are by default the **raw**
  values (`use_scaled_for_prompt=False`), because raw values are
  what domain experts and feature descriptions naturally talk
  about. Set `use_scaled_for_prompt=True` if you want the LLM to
  see standardised values instead — e.g. when raw scales are
  uninformative and only relative position matters.

### 13.3 The fit-time variance filter

`MolActAgent.fit` runs a one-shot variance filter *before* handing
data to the preprocessor, dropping any column whose variance is
below `variance_threshold` (default `0.0`, i.e. drop only
constants). Removed columns are also dropped from
`feature_names` and `feature_descriptions` so downstream prompts
stay consistent.

## 14. Stage 3 — Contextual similarity and neighbour selection

This is where the §6 commitment becomes math. The implementation
is in
[`molactagent/neighbors.py`](../molactagent/neighbors.py).

### 14.1 Definitions

Let `X` be the standardised reference matrix of shape `(n, d)`
and let `M = max_pairwise_distance(X)` be the largest off-diagonal
distance, computed once at fit-time. (`M` is what we normalise
distances by so the resulting similarities live in `[0, 1]`.)

For a sample `a`, define its **k_local-neighbourhood signature**
as the indices of its `k_local` closest reference samples (using
the same metric as the agent — Euclidean by default) together with
their distances. We write `sig(a) = (idx, dist)`. By default
`k_local = 3`. For reference samples this is precomputed at fit
time; for the test sample it is computed on the fly.

Then:

- **Feature similarity** between samples `a` and `b`:

  ```
  feat_sim(a, b) = max(0, 1 - distance(a, b) / M)
  ```

  This is just inverse normalised distance, clipped to non-negative.

- **Neighbourhood similarity**: take the cross-distance matrix
  between `sig(a).indices` and `sig(b).indices` (using the
  *reference points*, not the signature distances) and form

  ```
  d_ab = mean over rows of min over columns
       = "average distance from each of a's neighbours to its
          closest of b's neighbours"
  d_ba = mean over columns of min over rows
       = the symmetric counterpart
  avg  = 0.5 * (d_ab + d_ba)

  nbr_sim(a, b) = max(0, 1 - avg / M)
  ```

  This is the symmetric mean min-distance between the two
  neighbourhoods, normalised.

- **Contextual similarity** is the blend:

  ```
  ctx_sim(a, b) = b * feat_sim(a, b) + (1 - b) * nbr_sim(a, b)
  ```

  with `b = similarity_blend`, default `0.5`. Both components
  live in `[0, 1]`, so the blend does too.

- **Contextual distance** is `1 - ctx_sim`, used for ranking.

### 14.2 Per-class top-k ranking (classification path)

`NeighborFinder.find_local_references(X_ref_scaled, y_ref, x_test_scaled)`
returns a `NeighborResult` containing, for every class with at
least one reference, the indices of the top-k references *ranked
by contextual similarity to the test sample*. (Falls back to fewer
than k when the class has fewer references.)

For each class `c`:

1. Mask the reference indices where `y_ref == c`.
2. Compute `ctx_sim(x_test, ri)` for every `ri` in the mask
   (one cross-distance matrix per call).
3. Rank by descending similarity (equivalently ascending
   contextual distance).
4. Take the top `k` (or all of them if fewer exist).
5. Record raw feature distances too — they are kept around so the
   tuple-level "degenerate reference" check (where a reference is
   numerically identical to the test sample) keeps working
   independently of the contextual blend.

The result also flags `class_degenerate[c]`: a boolean array
marking which picked references are within `degenerate_eps` (1e-6
by default) of the test sample in raw feature distance.

### 14.3 Flat top-k (regression / CCR path)

`NeighborFinder.find_top_k_references(X_ref_scaled, x_test_scaled, k)`
ignores classes and returns the top `k` references overall, ranked
by contextual similarity. CCR uses this internally to score every
reference once per phase, then filters by the winning class.

### 14.4 The `NeighborhoodSignature` cache

Reference signatures are precomputed once at `fit` time; the
test-sample signature is computed on demand inside the per-sample
similarity helper. The cache is what makes contextual similarity
inexpensive at predict time — the only per-call cost is one
distance vector from the test sample to each reference plus a
small amount of work per kept reference.

### 14.5 A worked numeric example

Suppose the standardised reference matrix is:

```
R0 = [ 0.0,  0.0]
R1 = [ 0.1, -0.1]
R2 = [-0.1,  0.1]
R3 = [ 5.0,  5.0]
R4 = [ 5.1,  5.1]
R5 = [ 4.9,  4.9]
```

with classes `[A, A, A, B, B, B]`. Set `k_local = 2`. The
pairwise distance matrix (Euclidean) makes `R0`'s 2-NN signature
`{R1, R2}` (distances ~0.14 each) and `R3`'s `{R5, R4}` (distances
~0.14 each). The largest pairwise distance `M ≈ 7.21` (between
`R2` and `R4`).

A test sample at `x = (0.0, 0.05)`:

- Feature distance to `R0`: ~0.05; to `R3`: ~7.07.
- Test signature: 2-NN are `R0` and `R2` (dist ~0.05 and ~0.11).
- `feat_sim(x, R0) ≈ 1 - 0.05/7.21 ≈ 0.993`.
- `nbr_sim(x, R0)` between `{R0, R2}` and `{R1, R2}`: every
  reference in one neighbourhood has a counterpart in the other
  with distance ≤ 0.2, so `nbr_sim ≈ 1 - 0.07/7.21 ≈ 0.990`.
- `ctx_sim(x, R0) = 0.5*0.993 + 0.5*0.990 ≈ 0.992`.

`R3`'s contextual similarity to `x` will be small from both
components — `R3` is far from `x` in feature space, *and* `R3`'s
neighbourhood (`{R4, R5}`) is far from `x`'s neighbourhood
(`{R0, R2}`). The contextual blend pushes both signals in the
same direction; that is the typical case.

The interesting cases are the ones where the two components
disagree — a sample that is numerically close but structurally
elsewhere. In those cases the blend reduces the similarity
relative to a pure-feature ranking, which is exactly the design
intent.

## 15. Stage 4 — Adaptive top-M class selection

For classification, MolActAgent does not always compare every class
against every other class. It picks the M classes "in contention"
for the test sample — typically M=2, sometimes M=3, rarely M=4.

The selector is `NeighborFinder.select_top_m_classes(result)`:

1. Sort classes by `min_distance(c)` ascending. (This is the
   smallest contextual distance among that class's k-nearest
   references.)
2. Drop any class with no references (distance `∞`).
3. If two or fewer classes remain, return them all.
4. Compute the floor: `floor = max(closest_d, eps) * m_threshold`.
   Default `m_threshold = 1.5`.
5. Keep every class whose closest distance is `≤ floor`.
6. Guarantee at least 2: if step 5 only kept one, fall back to
   the two closest.
7. Cap at `max_m`: default 4.

The `eps` term protects against the degenerate case where the
closest class is at distance ~0 — without it, every other class
would be excluded because `0 * 1.5 = 0`.

#### Empirical behaviour

On real datasets we have measured the M distribution sits roughly
at:

- M=2 in ~80% of test samples.
- M=3 in ~19%.
- M=4 in the rare remaining ~1%.
- M=1 never (the floor in step 6 prevents it).

The M=3 cases are exactly the ones where the test sample sits
near a class boundary, so the LLM gets to see *all three nearby
classes* and reason about which one fits — a graceful upgrade
rather than a separate code path.

## 16. Stage 5 — Per-tuple classification

This is the per-tuple inner loop, in `_classify_tuple` in
[`molactagent/agent.py`](../molactagent/agent.py). It is run for
every `(test, ref_1, ref_2, …, ref_M)` tuple where each `ref_i` is
drawn from a different candidate class's neighbour pool. Tuples
are formed by Cartesian product over the per-class neighbour lists
in `_predict_one_classification`; degenerate tuples (a chosen
reference is numerically identical to the test sample within
`atol=1e-6`) are dropped before evaluation.

### 16.1 Discriminative-feature mask

For a tuple of M reference vectors, the discriminative mask picks
features whose values are not all identical across the M
references:

```
disc_mask[j] = (ref_1[j] != ref_2[j]) OR (ref_1[j] != ref_3[j]) OR ...
            for any pair within the M references
```

Implemented as `(ref_stack != ref_stack[0]).any(axis=0)` in NumPy.
Non-discriminative features are dropped from this tuple's prompt;
they cannot inform the contrast and would only consume tokens.

If `n_disc == 0` (rare — happens when references are exactly
equal), the tuple short-circuits to an `Unknown` verdict with
zero-confidence and a clear "no comparison possible" reason.

### 16.2 Per-feature alignment tally and baseline class

For each discriminative feature, ask: which class's reference is
numerically closest to Test on this feature? That class earns a
"vote" on this feature. If two or more classes are equidistant
(rare but possible), the feature contributes to a separate
**neutral** bucket rather than tying the vote — see lines 654-666
of `agent.py`.

Concretely:

```
abs_diffs[c, j] = |ref_c[j] - test[j]|
min_d[j]        = min over c of abs_diffs[c, j]
n_aligned[c]    = number of j where abs_diffs[c, j] == min_d[j]
                  AND only one c achieves this minimum
n_neutral       = number of j where ≥ 2 classes tie at min_d[j]
```

Sums obey `sum(n_aligned[c]) + n_neutral == n_disc`.

The **baseline class** is the argmax of `n_aligned`:

- If all counts are zero → `Unknown`.
- If the top two counts are tied → `Unknown` (let the LLM break
  the tie with a domain-grounded argument).
- Otherwise → the unique top class.

This is the deterministic, embarrassing-to-get-wrong number from
§5: a class identified by raw numerical similarity. The LLM is
told the tally in the prompt and asked to default to the baseline
unless it can decisively contradict it.

### 16.3 Per-reference texture blocks

For each reference in the tuple, retrieve its precomputed
k_local-neighbourhood signature (§14.4) and assemble a small block
showing the reference together with its k_local nearest reference
neighbours and *their* class labels (or, for regression's CCR
phases, their target values). The block is rendered by
`PromptBuilder.format_reference_neighbourhood_block` as an aligned
text table:

```
=== A_R12 (class A) ===
Identifier  |    Value | feat_1 | feat_2 | ...
---------------------------------------------------
A_R12       |        A |   1.23 |   4.56 | ...
R7          |        A |   1.20 |   4.55 | ...
R9          |        B |   1.40 |   4.60 | ...
R3          |        A |   1.18 |   4.50 | ...
```

Reading downward, the LLM perceives the local "neighbourhood
composition" (cluster interior vs boundary) without any
"purity score" being computed in Python. The prompt's preface
calls this out: *"Treat boundary references as weaker evidence
than territory-deep ones."* This is the §7 commitment in action.

### 16.4 Prompt construction

`PromptBuilder.build_tuple_prompt` assembles the full per-tuple
prompt. It contains, in this order:

1. **System message** declaring the analyst persona and listing
   the candidate classes for this comparison.
2. **Side-by-side table** of the discriminative features only,
   one column per class reference plus a Test column.
3. A **filter note** stating how many of the original features
   are discriminative.
4. **Feature descriptions** (when provided), restricted to the
   discriminative subset.
5. **DETERMINISTIC ALIGNMENT TALLY** block, surfaced as ground
   truth: per-class counts, neutral count, and the baseline
   verdict the LLM is expected to match.
6. **PER-REFERENCE LOCAL CONTEXT** blocks (§16.3).
7. **Task description** (the reasoning steps to perform
   internally).
8. **Output format** specification — a JSON schema with
   `label`, `confidence`, `key_drivers` (3-6 entries with
   per-driver `feature` / `favored_class` / `strength` /
   `test_value` / `favored_value` / `why`), `supporting_features`,
   `opposing_features`, `summary`, `counter_evidence`,
   `uncertainty`.
9. **Rules** restating the most important constraints (JSON only,
   match the baseline unless decisively contradicted, etc.).

The system message is permanent across retries; the user message
is what gets re-asked when validation fails (§17).

### 16.5 Parallel execution

Within a single test sample, all tuples are evaluated in parallel
via a `ThreadPoolExecutor` capped at `min(max_workers, n_tuples)`
(default `max_workers = 8` from the pipeline; the inner agent's
own default is 100). Per-tuple results are collected in input
order so downstream aggregation is reproducible regardless of
completion order.

Tuple failures (LLM exception, timeout) become "fallback" tuple
records with `label=Unknown`, `confidence=0`, and an explanatory
`reason` field. Throttle events are tracked per call (§20) and a
warning is emitted at the end of the sample if the LLM throttled
us.

### 16.6 Tuple result schema

A successful per-tuple result is a dict with these fields:

| Field | Type | Description |
|---|---|---|
| `candidate_classes` | `list[str]` | The M classes contrasted in this tuple. |
| `ref_indices` | `dict[str, int]` | For each candidate class, the reference index used. |
| `label` | `str` | The verdict — one of the candidate classes or `"Unknown"`. |
| `confidence` | `float` | The LLM's stated 0-1 confidence. |
| `class_scores` | `dict[str, float]` | Computed deterministically from `n_aligned / total_aligned` (or uniform if no votes). NOT from the LLM. |
| `n_aligned` | `dict[str, int]` | Per-class numerical-alignment counts. |
| `n_neutral` | `int` | Features where ≥ 2 classes were equidistant. |
| `baseline_class` | `str` | Determined from `n_aligned` argmax. |
| `summary` | `str` | LLM's 2-3-sentence narrative. |
| `reason` | `str` | First 200 chars of the summary, kept for terse displays. |
| `discriminative_features` | `list[str]` | Names of the features in the prompt's table. |
| `key_drivers` | `list[dict]` | Validated, deduped, in-strength order. Each entry has `feature`, `favored_class`, `strength`, `test_value`, `favored_value`, `why`. |
| `supporting_features` | `list[str]` | LLM-cited features supporting the verdict. |
| `opposing_features` | `list[str]` | LLM-cited features opposing the verdict. |
| `counter_evidence` | `str` | The LLM's frank discussion of what would overturn the verdict. |
| `uncertainty` | `str` | Single-sentence note on remaining uncertainty. |

## 17. Stage 6 — Validation, override policy, retry

The LLM's per-tuple JSON is *parsed* and *validated* by Python
before being accepted.

### 17.1 Parsing

`_safe_parse_tuple_json(raw)` extracts the JSON object from the
LLM's response:

1. Strip a `\`\`\`json` ... `\`\`\`` fence if present.
2. Try `json.loads` directly.
3. If that fails, find the first balanced `{ ... }` block and try
   `json.loads` on that.
4. If still no valid JSON, return `None` and the validator will
   issue a "your output was not valid JSON" message.

### 17.2 Rule-based validation

`_check_tuple(parsed, disc_set, candidate_classes, n_aligned, baseline_class)`
runs these checks:

- `label` must be one of the candidate classes or `"Unknown"`.
- `confidence` must coerce to a float in `[0, 1]`; clamped if
  out of range, defaulted to `0.5` if missing/unparseable.
- `key_drivers` must be a list with ≥ 2 entries; each entry must
  cite a feature that exists in the discriminative set; each
  entry's `favored_class` must be one of the candidates;
  duplicate features are deduped; `strength` is clamped to
  `[0, 1]`.
- `supporting_features` and `opposing_features` are coerced to
  lists of strings.
- `summary`, `counter_evidence`, `uncertainty` are coerced to
  strings (defaulted to empty).

### 17.3 The soft baseline check (the override policy)

The crucial check is the soft baseline check — the
implementation of the §5 commitment that the LLM may override the
witness only with decisive evidence:

```
If the LLM's label disagrees with baseline_class, AND both labels
are real candidate classes (not Unknown), then:

  same_dir = sum of strengths over drivers favouring baseline_class
  opp_dir  = sum of strengths over drivers favouring the LLM's label
  override_strong = (opp_dir > same_dir + 1.0)

  If NOT override_strong:
    add a "your drivers do not justify overriding the baseline"
    issue and ask the LLM to either set its label to baseline_class
    OR strengthen the drivers favouring its current label.
```

The `+ 1.0` margin is a witness-level numerical threshold. It is
deliberately blunt: we want to reject "the LLM picked a different
label but gave drivers that aren't actually decisive", not
adjudicate close calls.

### 17.4 Retry loop

If `_check_tuple` flags issues, the agent issues a follow-up
message asking the LLM to fix them and retries. The maximum
number of attempts is `max_check_retries + 1` (default 4 total
attempts: the first try + 3 retries). On each retry, the prompt
includes the LLM's previous response and a bullet list of the
specific issues — see `PromptBuilder.build_feedback_messages`.

If all retries are exhausted, the agent does **not** drop the
tuple; instead it falls back to the deterministic baseline:

```
label       = baseline_class
class_scores = n_aligned / total_aligned    (or uniform if all zero)
margin      = top_score - runner_score
confidence  = 0.5 * margin + 0.4
key_drivers = []   (no LLM rationale was usable)
reason      = "Validation failed N times; used deterministic baseline."
```

This way every tuple contributes useful information to the
aggregation — the system never silently fails on a single bad
LLM call.

### 17.5 Final clamp in `_finalize_tuple`

Even if the LLM's JSON passes validation, the final tuple record
clamps `label` to `baseline_class` whenever both are real
candidates and they disagree. This is the last-resort safety net
described in §5.3 — the witness wins by default. The LLM's
reasoning still feeds the rationales, drivers, and explanation,
but the per-tuple verdict is anchored.

## 18. Stage 7 — Aggregation, calibration, attribution

After all tuples for a test sample have been evaluated,
`_aggregate_results` produces the per-sample verdict.

### 18.1 Vote aggregation

For each candidate class `c`, count tuples whose `label == c`
(weighted by their `confidence`):

```
class_votes[c]   = number of tuples where label == c
class_weights[c] = sum of tuple confidence over those tuples
n_unknown        = number of tuples where label not in any class
n_decisive       = sum over c of class_votes[c]
```

If `n_decisive == 0` → `final_label = Inconclusive`.

Otherwise sort by `class_weights` descending, take the winner and
runner-up, and apply the **decisive margin** rule:

```
threshold = 0.05 * max(n_decisive, 1)
if winner_w > runner_w + threshold:
    final_label = winner
    margin = (winner_w - runner_w) / max(winner_w + runner_w, eps)
else:
    final_label = Inconclusive
    margin = 0
```

The `0.05 * n_decisive` bias keeps a single low-confidence vote
from deciding a tie — to win, the leader must outweigh the
runner-up by more than 5% of the decisive-tuple count.

### 18.2 Confidence calibration

The final confidence is a multiplicative blend of three factors,
matching the binary calibration formula generalised to multi-class:

```
margin_score      = 0.5 + 0.5 * clamp(margin, 0, 1)
sample_size_score = min(1, smallest_candidate_class_size / 10)
coverage_score    = n_decisive / max(n_total_tuples, 1)

confidence = margin_score * (
    0.5 + 0.25 * sample_size_score + 0.25 * coverage_score
)
confidence = clamp(confidence, 0, 1)
```

Interpretation:

- `margin_score ∈ [0.5, 1]` captures *how far ahead* the winner
  is. A perfectly lopsided verdict (margin 1) maps to 1; a tie
  (margin 0) maps to 0.5.
- `sample_size_score ∈ [0, 1]` discounts confidence when the
  smallest candidate class has fewer than 10 reference samples.
- `coverage_score ∈ [0, 1]` discounts confidence when many
  tuples were inconclusive.

For the `Inconclusive` final label, the formula collapses to
`max(0, 0.4 - 0.2 * margin)` — bounded above by 0.4 to reflect
that a non-verdict is *not* a high-confidence statement.

### 18.3 Driver attribution

`_compute_driver_attribution(tuple_results, feature_descriptions)`
aggregates only features that the LLM **cited as drivers** —
features merely present in the discriminative set but not chosen
as drivers contribute zero to the attribution.

For each driver entry across all tuples:

```
per_feature[fn].n_cited                 += 1
per_feature[fn].class_strength[favored] += strength
per_feature[fn].rationales              += [why]
```

After ingestion, for each cited feature:

```
sorted_class_strength = sort cs_dict descending by strength
top_strength    = sorted[0].strength
runner_strength = sorted[1].strength if any else 0
net             = top_strength - runner_strength
total           = sum over classes of strength
dominant_class  = top_class if net > 0.05 else "Mixed"
```

The output is a list of `attribution` records:

```
{
  "feature": <name>,
  "description": <from feature_descriptions or "">,
  "n_cited": <count>,
  "class_strength": {<class>: <round(strength, 3)>, ...},
  "weighted_score": <round(net, 3)>,        # signed magnitude
  "abs_strength": <round(total, 3)>,        # absolute total
  "dominant_class": <class or "Mixed">,
  "rationales": <up to 3 example "why" strings>
}
```

Sorted by `(abs_strength, n_cited)` descending — features cited
most decisively are first.

### 18.4 Per-class top features

A six-feature-per-class capped list, derived from the attribution
above, is exposed as `key_features_per_class` for use in reports.
Only features whose `dominant_class` matches the class are
included.

### 18.5 Aggregated result schema

The dict returned from `_aggregate_results` contains:

| Field | Type | Description |
|---|---|---|
| `test_index` | `int` | Position in the test set. |
| `task` | `str` | `"classification"`. |
| `final_label` | `str` | The verdict. |
| `confidence` | `float` | Calibrated 0-1 confidence. |
| `confidence_basis` | `dict` | `margin_score`, `sample_size_score`, `coverage_score`, `rationale`, `basis_text`. |
| `reason` | `str` | Vote tally + rationale for the human reader. |
| `supporting_evidence` | `str` | Top features dominantly favouring the verdict. |
| `opposing_evidence` | `str` | Top features favouring other classes. |
| `uncertainty` | `str` | Frank statement of remaining uncertainty. |
| `key_features_per_class` | `dict[str, list[str]]` | Per-class top driver features. |
| `tuple_results` | `list[dict]` | The per-tuple records. |
| `driver_attribution` | `list[dict]` | Aggregated driver attribution. |
| `vote_summary` | `dict` | `total`, per-class `votes` and `weighted`, `num_unknown`, `n_degenerate`. |
| `candidate_classes` | `list[str]` | The M classes contrasted. |
| `reference_info` | `dict` | Reference set sizes, neighbour counts, comparison count, etc. |
| `test_feature_values` | `dict[str, float]` | The test sample's raw (or scaled) features by name. |

This schema is what `report.py` consumes and what the optional
`generate_explanation` LLM call expands into a Markdown narrative.

## 19. The CCR regression backend

When `task == "regression"`, `MolActAgent.predict_one` delegates
to `CrossClassificationRegressor` (CCR), in
[`molactagent/cross_classification.py`](../molactagent/cross_classification.py).
This section explains CCR end to end.

### 19.1 The big picture

CCR turns regression into two parallel multi-class
classifications:

- **Phase 1 — target view.** Bin the continuous target into N
  equal-frequency bins (`bin_0` … `bin_{N-1}`), then run
  MolActAgent classification over those bins.
- **Phase 2 — feature view.** Cluster the standardised features
  into M cosine-distance / average-linkage clusters
  (`fc_0` … `fc_{M-1}`), then run MolActAgent classification over
  those clusters.

Each phase yields a winning class (a target bin or a feature
cluster) and a confidence. CCR then picks the top-`k_per_class`
references in each winning class via contextual distance, dedupes
by reference index, and combines the resulting target values into
a confidence-weighted numeric prediction.

The classification engine inside CCR is *the same*
`MolActAgent` used for the standalone classification path —
adaptive top-M=2 contrastive tuples, deterministic alignment
tally, soft override policy, retry loop. CCR is fully insulated
from the per-tuple regression LLM prompts the older bootstrap-tuple
backend used; it has no regression-specific prompts.

### 19.2 Hyperparameter resolution

`resolve_n_classes(n_full, mode)` sets N (target bins) and M
(feature clusters) given the dataset size. The default mode is
`"sqrt"`:

```
N = max(2, min(floor(sqrt(n_full)) + 1, n_full - 1))
```

For example: `n=24 → N=5`, `n=100 → N=11`, `n=10000 → N=101`.

The `+ 1` is a small bias toward more granular bins/clusters
because empirically (BSLT n=16, penguins n=24) more clusters
helped without proportionally hurting per-phase classification
cost — top-M=2 caps each multi-class call to a contrast between
the two closest classes regardless of how many classes exist in
total.

Other modes:

| Mode | Formula | Use |
|---|---|---|
| `"sqrt"` | `floor(sqrt(n)) + 1`, clamped to `[2, n-1]`. | **Default**, preferred for new runs. |
| `"sqrt-floor"` | `floor(sqrt(n))`, clamped to `[2, n-1]`. | Slightly more conservative. |
| `"sqrt-capped-10"` | `floor(sqrt(n))`, clamped to `[2, 10]`. | Legacy; reproduces older runs. |
| `"fixed"` | `fixed_value`, clamped to `[2, n-1]`. | Manual override. |

`n_target_bins` and `n_feature_clusters` can each be set
explicitly to override the per-phase value.

### 19.3 Equal-frequency target binning

`equal_frequency_bins(y, n_bins)` uses `pd.qcut` with
`duplicates="drop"`. The result is bins of approximately equal
**count** rather than equal **width** — robust against skewed
targets where a few large outliers would otherwise dominate
width-based bins.

Bin labels are strings `"bin_0"` (lowest target values) through
`"bin_{N-1}"` (highest), so the LLM can reason about ordinal
position.

If `qcut` cannot produce N usable bins (e.g. heavy ties), the
implementation falls back to `np.linspace` cuts. If even that
fails to assign every row a bin, it returns a single-bin label
array as a degenerate fallback.

### 19.4 Cosine feature clusters

`cosine_feature_clusters(X, n_clusters)`:

1. Standardise the features (`StandardScaler`).
2. Compute the cosine distance matrix.
3. Run scikit-learn's `AgglomerativeClustering` with
   `metric="precomputed"`, `linkage="average"`,
   `n_clusters=n_clusters_eff` (effective number, clamped to
   `[1, n-1]`).
4. Return string labels `"fc_0"` through `"fc_{M-1}"`.

Standardisation matters: raw cosine distance has no useful meaning
when features live on wildly different scales, which is the
common case in tabular data with mixed-magnitude columns.

### 19.5 Per-test-sample prediction

`CrossClassificationRegressor.predict_one(x_test)`:

```
result_t = agent_target.predict_one(x_test)        # Phase 1
winner_t = result_t.final_label
       (or, if Inconclusive/Unknown, the class with highest
        weighted vote in result_t.vote_summary; otherwise None)
nn_t     = top k_per_class refs in winner_t,
           ranked by contextual distance to x_test

result_f = agent_feature.predict_one(x_test)       # Phase 2
winner_f, nn_f = same as above for feature-cluster classes

merged   = dedupe by reference index, keeping the smaller
           contextual distance whenever a reference is picked
           by both phases

if merged is empty:
    return _build_refusal(...)

if aggregator in {"conf", "conf-distance"}:
    accumulate per-reference weights:
       weight(r) = sum over phases that picked r of:
                   (phase_conf)              if "conf"
                   (phase_conf / (d + eps))  if "conf-distance"
    return sum(target[r] * weight(r)) / sum(weight(r))

else:
    if aggregator == "median":   return median(targets)
    if aggregator == "weighted": return weighted-mean by 1/(d+eps)
    else:                         return mean(targets)
```

The aggregator dispatch and refusal policy live in `predict_one`
(the path that produces a numeric prediction) and `_build_refusal`
(the path that returns `predicted_value=None`).

### 19.6 The refusal policy

CCR returns `predicted_value=None` ("REFUSED") when:

1. Both phases produced no usable winner class. (This is rare —
   it would mean the inner MolActAgent could not even pick a
   weighted-vote leader.)
2. The confidence-weighted aggregator returned no usable weight.
3. A simple aggregator returned a non-finite value.

The refusal record has the same shape as a normal result so
downstream code (`pipeline.py`, `report.py`) handles refusals
uniformly. The pipeline's per-sample row shows `predicted_label =
"REFUSED"` and the explanation report says so plainly.

### 19.7 Confidence and the empirical interval

For a non-refused prediction:

```
present_confs = [phase_conf for each phase that produced a winner]
confidence    = mean(present_confs) if any else 0
```

We mean over phases that *produced a winner*, not over all phases:
if Phase 2 refused but Phase 1 succeeded, averaging in 0 would
artificially halve Phase 1's confidence.

The `interval` reported is the empirical [min, max] of the picked
references' actual target values — a tight, *honest* bracket
rather than a parametric interval. It is what the report's dot
plot visualises.

The `extrapolation_flagged` field is `True` when the prediction
falls strictly outside the training-target range. This is a
diagnostic for the report; it does not change the numeric
prediction.

### 19.8 Driver attribution synthesis

`_synthesize_driver_attribution(result_t, result_f, X_train, y_train, ...)`
combines the two phases' classification-style driver attributions
into a regression-flavoured list:

1. Merge both phases' driver entries by feature name; sum
   `n_cited` and `abs_strength`; track which phases cited each.
2. For each cited feature, compute the Pearson correlation
   between that feature column and the continuous target across
   the training set.
3. Set `dominant_direction`:
   - `"increases"` if `corr > 0.15`,
   - `"decreases"` if `corr < -0.15`,
   - `"mixed"` otherwise.
4. Sort by `(abs_strength, n_cited)` descending and keep the top
   12.

The Pearson value goes into `weighted_score` (so the report can
display a signed strength). The 0.15 threshold is a small,
conservative floor that prevents weakly-correlated features from
being labelled with a misleading direction.

Pearson is computed manually inside the function to avoid pulling
in `scipy.stats` for a single number.

### 19.9 CCR result schema additions

On top of the regression-flavoured analogue of the classification
schema, CCR adds:

| Field | Description |
|---|---|
| `task` | `"regression"`. |
| `method` | `"CCR"`. |
| `predicted_value` | The numeric verdict, or `None` for refusals. |
| `interval` | `(min, max)` of the picked references' actual target values. |
| `predictions` | The picked references' actual target values, in `selected_indices` order. |
| `phase_target` | `{predicted_class, confidence, candidate_classes, final_label, neighbours}` for Phase 1. |
| `phase_feature` | Same for Phase 2. |
| `selected_indices` | Sorted list of unique reference indices picked across both phases. |
| `selected_targets` | Their actual target values. |
| `selected_distances` | Their contextual distances (smallest across the two phases when both picked the same reference). |
| `aggregator` | The aggregator name (`"conf"`, `"conf-distance"`, `"mean"`, `"median"`, `"weighted"`). |
| `phase_views_agree` | `True` when at least one reference was picked by both phases. |
| `extrapolation_flagged` | See §19.7. |
| `tuple_results` | Two phase rows, each carrying the phase mean target, the phase interval, the phase confidence, and a small per-phase `drivers` list (legacy schema-compatible). |

## 20. The LLM client

[`molactagent/llm.py`](../molactagent/llm.py) wraps an OpenAI-
compatible `client.chat.completions.create(...)` call. The wrapper
has three jobs:

1. **Retries with backoff.** On exceptions, retry up to
   `max_retries` times (default 10), with a per-attempt delay of
   `retry_delay * (attempt + 1)` (default `2.0` s base). Throttle
   exceptions get the delay doubled.
2. **Throttle detection.** A `_is_rate_limit_error` helper checks
   for HTTP 429 / 5xx / "rate" / "limit" markers in the exception
   string or `status_code`. Each detected throttle increments
   `throttle_count` so the agent can warn the user post-hoc.
3. **Per-call statistics.** Each successful call appends to a
   shared `_call_log` recording `call_id`, `tag`, elapsed time,
   prompt and completion tokens, prompt char count, response
   char count, truncation flag. Aggregate counters power
   `client.stats` — total tokens, total time, average call
   time, etc.

There is also a **`ping`** method, used at pipeline preflight, that
fires a 1-token round trip. It bypasses the retry loop so
credential / endpoint errors surface immediately and translates
them into a `kind` taxonomy: `"auth"`, `"model"`, `"network"`,
`"unknown"`. The pipeline turns these into actionable
`LLMConfigError` messages with concrete next steps.

The wrapper is provider-agnostic in practice. Anything that
accepts `model`, `messages`, `temperature`, `max_tokens` over an
OpenAI-style endpoint works — DeepSeek, Qwen, OpenAI, Mistral,
Anthropic-via-OpenRouter, Google Gemini's compat endpoint,
Ollama, vLLM, LM Studio.

## 21. Pipeline orchestration

[`molactagent/pipeline.py`](../molactagent/pipeline.py) glues
everything together. The high-level entry point is
`run_prediction(reference, test, label_column, **kwargs)`;
`run_classification` and `run_regression` are thin aliases that
also assert the auto-detected task matches.

### 21.1 Step-by-step

1. **Coerce inputs**: DataFrames stay; paths are loaded via
   `io.load_table`. Background and feature descriptions follow
   the same rule.
2. **Validate**: `io.validate_inputs` runs full validation. If
   `is_valid=False`, return early (or raise if
   `raise_on_validation_error=True`).
3. **Reconcile task**: the detected task from
   `validation.summary["task"]` is the source of truth. If the
   caller passed an explicit `task`, the call fails with a clear
   error when the two disagree.
4. **Prepare**: `io.prepare_for_agent` builds the numpy arrays.
5. **Build the LLM client** (or use the supplied one). When
   `verify_llm=True`, call `_preflight_llm` to do a 1-token
   ping; surface any failure as a friendly `LLMConfigError`.
6. **Create the output directory** (if `output_dir` is set);
   refuse if it points to an existing file.
7. **Fit the agent**: one `MolActAgent.fit(X_ref, y_ref, task)`
   call.
8. **Predict** each test sample one by one (the parallelism is
   inside each per-sample call, across tuples). For each:
   - Run `agent.predict_one`.
   - Optionally run `agent.generate_explanation` (one extra LLM
     call per sample).
   - Write per-sample artefacts (`.json`, `.md`, `.html`) to the
     output directory if `per_sample_reports=True`.
   - Append a `SampleOutcome` to the result list, capturing
     `predicted_label`, `confidence`, `predicted_value` (for
     regression), `interval`, `true_label`, `elapsed_s`,
     `error`, paths.
9. **Write summaries**: `summary.csv` (default on),
   `summary.json` (off by default), `interpretability.md` and
   `interpretability.html` (default on; zero extra LLM calls,
   purely renderered from the per-sample evidence).
10. **Return** a `PipelineOutput` with everything: validation
    record, outcomes, output directory, file paths, and LLM
    stats.

### 21.2 Failure modes the pipeline handles cleanly

- LLM credential / endpoint failure → `LLMConfigError` with
  hints (validation file checked, action items, raw error).
- Per-sample `predict_one` exception → that sample's outcome has
  `predicted_label="Error"`, `confidence=0`, `error=str(exc)`.
  Other samples are unaffected.
- Validation failure → early return with `outcomes=[]`. Logged
  with a multi-issue report.

## 22. Report rendering

[`molactagent/report.py`](../molactagent/report.py) handles all
output rendering. The functions split into three layers:

### 22.1 Per-sample HTML report

`render_html_report(result, *, explanation, class_labels, title)`
produces one self-contained HTML file with five sections:

1. **Hero** — verdict badge in the dominant class colour,
   confidence number, evidence-basis caption.
2. **Top Drivers** card — a table of the highest-attribution
   features, each showing the dominant class (with arrow), the
   citation count, and the net weighted strength.
3. **Narrative** — the LLM-written Markdown report converted to
   HTML by `_md_to_html` (the project's own Markdown-subset
   converter that handles ATX headings, lists, pipe tables,
   bold/italic/code, and horizontal rules).
4. **Robustness** card — a vote consistency bar with one segment
   per class plus an Inconclusive segment.
5. **Tech Details** appendix — the full result JSON, collapsed.

For regression, the badge shows the predicted value with units
inferred from the label name; the robustness card is replaced
with a target-range dot plot showing where the picked references
sit; the drivers table shows direction (`increases` / `decreases`
/ `mixed`) instead of dominant class.

The colour palette is fixed at module load:

```
green, red, blue, orange, purple, cyan, yellow, pink
```

Class N maps to `palette[N % len(palette)]`. `Inconclusive` /
`Mixed` / unknown classes always render as grey (`#8e8e93`).

### 22.2 Per-sample Markdown report

The Markdown report is generated by the LLM via
`agent.generate_explanation` (one extra call per sample) which
itself calls the appropriate `PromptBuilder.build_..._prompt`.
The prompt instructs the LLM to write a fixed-section report
(Executive Summary; Key Drivers table; Why These Drivers Matter;
Robustness Across Comparisons; Counter-evidence Considered;
Confidence and Caveats; Recommended Next Steps) — `report.py`
just wraps it in HTML when needed.

### 22.3 Aggregate interpretability report

`render_interpretability_text` / `_md` / `_html` build a single
report covering the entire test set:

- A header counting samples, the per-class verdict distribution,
  any error rows.
- A per-sample row showing `sample_id`, predicted label, true
  label (when available), confidence, and a one-line "headline
  reason" extracted from the per-sample result.
- For regression, an aggregate scatter of predicted vs true (when
  truth is available), plus the global target range.
- Aggregate driver attribution — features that drove decisions
  *across* the test set.

These are written to `interpretability.md` and
`interpretability.html` in the output directory and consume zero
extra LLM calls — all the evidence comes from the per-sample
`SampleOutcome` records that the pipeline already collected.

## 22.5 The v0.3 layered API surface (estimators / metrics / tools / pp / datasets)

§11.2 introduces the layered API at a high level; this section
gives each sub-package the same per-stage treatment that §12-22
gives the engine. None of these sub-packages adds any prediction
logic — they all delegate to the engine documented in §12-21
above. They exist to make the engine **discoverable**,
**composable** with the broader scientific-Python ecosystem
(scikit-learn pipelines, cross-validation, calibration tooling),
and **convenient** for the most common workflows.

### 22.5.1 Result wrappers (`molactagent.result`)

`ClassificationResult` and `RegressionResult` are typed wrappers
around the per-sample dicts produced by `MolActAgent.predict_one` /
`CrossClassificationRegressor.predict_one`. They are pure facades:

- `result.label`, `result.confidence`, `result.candidate_classes`,
  `result.proba(classes=None)`, `result.driver_attribution`,
  `result.tuple_results`, `result.test_feature_values`, ... —
  named-attribute access for every field a downstream consumer
  typically reaches for.
- `result.to_dict()` returns the engine's exact wire-format dict.
  This is the round-trip invariant that lets these wrappers slot
  into the existing `pipeline.py`, `report.py`, and JSON
  serialisation paths without any other code knowing they exist.

`wrap_result(raw)` is a factory that dispatches on `raw["task"]`
and returns the right subclass. Used internally by the
estimators; available publicly for users who hold raw dicts and
want the typed accessors.

### 22.5.2 Estimators (`molactagent.estimators`)

`MolActAgentClassifier` and `MolActAgentRegressor` are scikit-learn-
style wrappers. They follow the
`fit(X, y) -> predict(X) -> predict_proba(X) -> score(X, y)`
contract so they slot into `cross_val_score`, `GridSearchCV`,
`Pipeline`, and any other consumer that accepts a
`BaseEstimator`-shaped object.

The wrappers are deliberately thin:

- `fit(X, y)` constructs an internal `MolActAgent`, calls
  `agent.fit(X, y, task=...)`, captures `classes_`,
  `n_features_in_`, `feature_names_in_`. No new training logic.
- `predict(X)` runs `agent.predict(X)` (which goes through
  `predict_one` per row) and projects the per-sample result dicts
  to a string-array of labels (or a float-array of values, NaN
  for refusals).
- `predict_proba(X)` derives the per-class probability matrix
  from `vote_summary['per_class'][cls]['weighted']` (the same
  continuous score the LOOCV evaluation script has used since
  day one) — see `result.ClassificationResult.proba_vector` for
  the exact formula. Sums to 1 within FP epsilon.
- `predict_with_confidence(X)` exposes the engine's calibrated
  confidence (margin × sample-size × coverage; §18.2) as a
  separate axis from `predict_proba`. They answer different
  questions; both are useful; both are exposed.
- `predict_with_interval(X)` (regressor only) returns the
  empirical [min, max] of the picked references' actual targets;
  see §19.7.
- `predict_full(X)` returns the typed `ClassificationResult` /
  `RegressionResult` per sample for callers who want the rich
  evidence.
- `score(X, y)` returns mean accuracy (classifier) or R²
  (regressor) — the standard sklearn `*Mixin` contracts.

The estimators cache the most recent `(X, results)` pair so
calling `predict`, then `predict_proba`, then
`predict_with_confidence` on the same `X` costs only ONE round of
LLM calls. This matters because each engine call is a real LLM
call.

`get_params` / `set_params` round-trip mirrors sklearn's clone
contract so the estimators work cleanly inside `cross_val_score`
and `GridSearchCV`.

### 22.5.3 Metrics (`molactagent.metrics`)

Pure post-hoc computations. No engine state, no LLM calls. Two
typed `Report` dataclasses with `to_table()` / `to_dict()` /
`to_dataframe()` exporters:

- `classification_metrics(y_true, y_pred, *, proba=None,
  classes=None, positive_class=None, negative_class=None)` —
  polymorphic entry point. Auto-detects binary vs multi-class:
  binary returns the full suite (accuracy, balanced accuracy,
  MCC, macro F1, per-class precision / recall / F1, specificity,
  ROC AUC, PR AUC, confusion matrix, unknown rate); multi-class
  returns the multi-class subset (no AUC, no per-class
  precision / recall / specificity). `Inconclusive` /
  `Unknown` predictions count as wrong for accuracy and confusion
  matrix purposes.
- `regression_metrics(y_true, y_pred, intervals=None)` — MAE,
  RMSE, R², Pearson r, Spearman r, median absolute error,
  fraction-within-interval (when intervals are supplied), refusal
  count.
- `format_comparison_table(reports)` renders a multi-method
  fixed-width table identical to the legacy
  `scripts/run_loocv_metrics.format_metrics_table` output, so JSON
  artefacts in `data/` from the research-output era stay
  comparable.

### 22.5.4 Tools (`molactagent.tl`)

Downstream analysis. All four tools work uniformly across
MolActAgent estimators and ordinary scikit-learn estimators, so
head-to-head comparisons go through one API:

- `loocv(estimator, X, y, *, fold_workers, ...)` — leave-one-out
  CV. Returns a `LOOCVResult` with `metrics`, `fold_records`,
  `predictions`, `confidences`, `elapsed_s`. Auto-detects
  classification vs regression from `y` dtype.
- `kfold_cv(estimator, X, y, *, n_splits=5, ...)` — same
  contract, k-fold variant. Stratified for classification, plain
  for regression.
- `compare_baselines(molactagent_estimator, X, y, baselines=...,
  cv="loocv", ...)` — runs MolActAgent + a curated set of sklearn
  baselines (default for classification: KNN3, RF, LR, SVM;
  default for regression: KNN3, Ridge, RF; "all" expands to the
  full catalogue including XGB if installed). Returns a
  `BaselineComparison` with `metrics_table()` matching the legacy
  fixed-width script output. Failures in individual baselines are
  caught and logged so a single broken baseline does not bring
  down a five-method comparison.
- `confidence_calibration(y_true, y_pred, confidence, *,
  fold_records=None, bins=10)` — Brier, ECE, reliability curve.
  Polymorphic: pass three parallel arrays OR a `fold_records`
  list from `loocv` directly. Routes through the package-wide
  `_detect_task` so binary integer labels (0/1) are correctly
  handled as classification rather than mis-treated as regression.
- `aggregate_drivers(per_sample)` — cohort-level per-feature
  driver summary across a batch of predictions or fold records.
  Same shape as the engine's per-sample driver attribution but
  rolled up across many samples; preserves `dominant_class`
  (classification) or `dominant_direction` (regression).

### 22.5.5 Preprocessing (`molactagent.pp`)

Two families of helpers:

**Chemistry** — `ma.pp.chemistry`, `ma.pp.descriptors`,
`ma.pp.fingerprints`, `ma.pp.scaffold`. Convert SMILES into the
MolActAgent-ready `(X, feature_names, feature_descriptions)` shape
in one call:

- `smiles_to_features(smiles, y=None, *, descriptors=True,
  maccs=True, morgan=False, ...)` — the user-facing entry point.
  Returns a `FeatureBundle` carrying `X`, `y`, `feature_names`,
  `feature_descriptions`, `smiles`, `scaffolds`, `scaffold_ids`,
  `metadata`. Variance-filters fingerprint blocks up front so
  prompts stay compact.
- `descriptors.compute(mol)` / `compute_matrix(mols)` — 26 RDKit
  descriptors with curated descriptions (the same catalogue
  every project benchmark uses).
- `fingerprints.maccs(mol)` / `morgan(mol, ...)` plus
  `MACCS_KEY_DESCRIPTIONS` — the 167-key plain-English MACCS
  description dictionary, available without RDKit.
- `scaffold_split(smiles, *, test_size=0.2, random_state=42)` —
  scaffold-disjoint train/test split. Returns a `ScaffoldSplit`
  with `train_idx`, `test_idx`, `train_scaffolds`,
  `test_scaffolds`, `scaffold_ids` (the last is `GroupKFold`-
  ready).

**Tabular** — `ma.pp.tabular`. Wraps the existing
`io.validate_inputs` + `io.prepare_for_agent` into a single
`load_dataset(reference, test, label_column, ...)` call returning
a `TabularBundle` with `X_train, y_train, X_test, y_test,
feature_names, feature_descriptions, background, class_labels,
target_range, test_index, task, validation` — ready to drop into
either the high-level pipeline or the new estimator wrappers.

### 22.5.6 Datasets (`molactagent.datasets`)

Bundled-example accessors:

- `load_iris_flowers()` — 24-row 3-class classification.
- `load_penguins()` — 24-row regression on body mass; auto-drops
  the metadata-only `species` column from the bundled CSVs
  before validation, per the dataset README.
- `load_bslt_regression(*, descriptors=True, maccs=True,
  morgan=False)` — chemistry regression. Featurises the bundled
  SMILES via `ma.pp.smiles_to_features` and returns a
  `FeatureBundle`. Looks for `examples/bslt/bslt.csv` first,
  falls back to the legacy `data/BSLT_huigui.csv` for back-compat.

All loaders raise `FileNotFoundError` with an explicit
"the bundled examples ship in the source repository, not the
wheel install" message when the underlying files are not present.

---

# Part IV — Reproducibility and Configuration

## 23. Hyperparameter reference

The system has more knobs than a casual reader notices because
most defaults match the BSLT n=16 and penguins n=24 ablation
results and rarely need touching. The complete list:

### 23.1 Pipeline defaults (`run_prediction` keyword arguments)

| Argument | Default | Effect |
|---|---|---|
| `task` | `None` | When `None`, auto-detect from the label column. Pass `"classification"` / `"regression"` to override. |
| `k` | `3` | Per-class neighbours retained at the witness layer. |
| `metric` | `"euclidean"` | Distance metric for both feature and neighbourhood distances. |
| `scaler` | `"standard"` | One of `standard`, `minmax`, `none`. |
| `max_workers` | `8` | Parallel LLM calls per test sample (per-tuple parallelism). |
| `per_sample_reports` | `False` | Whether to make the extra LLM call that writes a per-sample explanation. |
| `per_sample_formats` | `("html", "md", "json")` | Subset of formats to emit per sample. |
| `write_summary_csv` | `True` | `summary.csv` per output directory. |
| `write_summary_json` | `False` | `summary.json` adds the validation block and LLM stats. |
| `write_interpretability` | `True` | Aggregate `interpretability.{md,html}` files. |
| `verify_llm` | `True` | Pre-flight 1-token ping before the run begins. |
| `raise_on_validation_error` | `False` | Convert validation failure into a raised `ValueError`. |

### 23.2 `MolActAgent` defaults (apply to both classification and CCR's inner classifiers)

| Argument | Default | Effect |
|---|---|---|
| `k` | `6` | Per-class neighbours retained. (Pipeline overrides to `3`.) |
| `metric` | `"euclidean"` | |
| `scaler` | `"standard"` | |
| `feature_names` | `None` | Auto-generated `f0`, `f1`, ... when missing. |
| `feature_descriptions` | `None` | |
| `background` | `None` | |
| `use_scaled_for_prompt` | `False` | Show raw values in the prompt. |
| `max_workers` | `100` | |
| `variance_threshold` | `0.0` | Drop columns with variance ≤ this at fit time. |
| `diversity_ratio` | `0.1` | Reserved; not currently consumed by the neighbour finder. |
| `random_state` | `None` | Threaded through to the inner classifiers; reserved for future stochastic stages. |
| `max_check_retries` | `3` | Up to 3 retries (4 total attempts) for LLM JSON validation. |
| `llm_temperature` | `0.0` | Default LLM sampling temperature. |
| `m_threshold` | `1.5` | Adaptive top-M threshold (see §15). |
| `max_m` | `4` | Cap on M. |
| `k_local` | `3` | Per-reference neighbourhood size (texture). |
| `similarity_blend` | `0.5` | Feature/neighbourhood blend weight. |
| `ccr_aggregator` | `"conf"` | Confidence-weighted aggregation. |
| `ccr_k_per_class` | `2` | Top-k references per CCR phase. |
| `ccr_agent_k` | `2` | Inner MolActAgent's `k` inside CCR phases. |
| `ccr_n_mode` | `"sqrt"` | Phase class-count mode (see §19.2). |
| `ccr_n_target_bins` | `None` | Manual override for Phase 1. |
| `ccr_n_feature_clusters` | `None` | Manual override for Phase 2. |

### 23.3 LLM client defaults (`LLMClient` constructor)

| Argument | Default |
|---|---|
| `temperature` | `0.0` |
| `max_tokens` | `2000` |
| `timeout` | `120.0` (seconds) |
| `max_retries` | `10` |
| `retry_delay` | `2.0` (seconds, base) |

### 23.4 What to tune, when

- **Smaller datasets**: leave `k=3` or even `k=2`. CCR's per-phase
  defaults already adapt via `floor(sqrt(n))+1`.
- **Many classes (> 5)**: `m_threshold` may need lowering (e.g.
  `1.3`) so M stays at 2 more often and prompts stay short.
  Raise `max_m` only if you genuinely want > 4 classes contrasted
  in a single tuple.
- **Big LLM costs**: turn off `per_sample_reports`. The aggregate
  `interpretability.{md,html}` files are zero extra LLM calls.
- **Throttling / rate limits**: lower `max_workers`. The throttle
  warning in the per-sample log will tell you when to act.
- **Highly skewed regression targets**: stick with the
  equal-frequency binning (Phase 1's binning is `qcut`-based and
  already handles skew); consider `n_target_bins` overrides only
  for very small datasets where the default `floor(sqrt(n))+1`
  produces too few bins.

## 24. LLM configuration

`load_llm_config` resolves connection settings from three sources
with this precedence (highest to lowest):

1. Explicit keyword arguments to `load_llm_config(...)`.
2. Environment variables: `MOLACTAGENT_API_KEY`,
   `MOLACTAGENT_BASE_URL`, `MOLACTAGENT_MODEL`.
3. The supplied config file (or, if not given, the first file
   matching `DEFAULT_CONFIG_FILENAMES` found by walking up from
   the current directory):
   - `llm_config.jsonc` (recommended)
   - `llm_config.json`
   - `llm_config.yaml` / `llm_config.yml`
   - `LLM_API.txt` / `llm_api.txt` (legacy)

The required fields are `api_key`, `base_url`, and `model_name`.
Optional fields (`temperature`, `max_tokens`, `timeout`,
`max_retries`) inherit the `LLMClient` defaults when not given.
Field-name aliases are normalised: `apiKey`, `baseUrl`,
`Modelname`, etc., all map to the canonical key.

### 24.1 Multi-profile config files

JSONC and YAML configs can carry multiple provider blocks (an
array of objects, or a top-level `profiles` key). `_select_profile`
picks one based on the optional `profile` argument:

- `None` / `"last"` / `"auto"` / `"default"` → last profile in the
  file. (The "last is the active provider" convention is what the
  starter JSONC implies.)
- `"first"` → first profile.
- `"deepseek"` or `"ds"` → first profile whose URL or model name
  contains `deepseek`.
- `"qwen"` or `"dashscope"` → likewise for Qwen / DashScope.
- Anything else → fall back to a substring match against
  `model_name`.

### 24.2 Starter file

`molactagent init-config` writes a starter JSONC (default) or
legacy `LLM_API.txt`. The JSONC contains an active DeepSeek block
plus commented-out templates for OpenAI, Qwen, Anthropic via
OpenRouter, Gemini's OpenAI-compat endpoint, Mistral, and a
local Ollama / vLLM example. The user uncomments one block at a
time.

### 24.3 Build the client

`build_llm_client(LLMConfig)` is a one-liner: it instantiates
`LLMClient(**config.to_client_kwargs())`. The pipeline calls this
when no `llm_client=` is supplied.

## 25. Building the system from scratch

Below is the minimum you need to reproduce MolActAgent from
scratch. Each step references the canonical implementation.

### 25.1 Choose the dependencies

- `numpy>=1.24`, `pandas>=2.0` for arrays / dataframes.
- `scikit-learn>=1.3` for `StandardScaler` / `MinMaxScaler`,
  `pairwise_distances`, `AgglomerativeClustering`.
- `openai>=1.0` for the OpenAI-compatible LLM client. (Any
  HTTP wrapper would do; the project chose `openai` because every
  major provider exposes an OpenAI-compatible endpoint.)
- Optional: `pyyaml` for YAML config support, `openpyxl` for
  Excel, `pyarrow` for Parquet.

### 25.2 Implement the witness layer

```
class NeighborFinder:
    def fit(X_ref):
        store X_ref
        compute and store full pairwise distance matrix (N x N)
        store max_distance = max(off-diagonal entries), default 1.0
        for each reference i, store the indices and distances of
            its k_local nearest *other* references (its signature)

    def feature_similarity(a, b):
        return max(0, 1 - distance(a, b) / max_distance)

    def neighbourhood_similarity(sig_a, sig_b):
        compute cross-distance matrix between the reference points
            indexed by sig_a and sig_b
        d_ab = mean over rows of the column-wise minimum
        d_ba = mean over columns of the row-wise minimum
        avg  = (d_ab + d_ba) / 2
        return max(0, 1 - avg / max_distance)

    def contextual_similarity_to_test(x_test, ref_idx):
        feat = feature_similarity(x_test, X_ref[ref_idx])
        nbr  = neighbourhood_similarity(test_sig(x_test),
                                        signature(ref_idx))
        return blend * feat + (1 - blend) * nbr

    def find_local_references(X, y, x_test):
        for each unique class c in y:
            score every reference of class c by ctx_sim to x_test
            keep top k (or fewer if class is small)
            record raw feature distance separately
        return per-class indices, ctx-distances, and raw feature
        distances

    def select_top_m_classes(result, m_threshold=1.5, max_m=4):
        sort classes by min ctx-distance ascending
        keep classes whose min distance ≤ threshold * closest
        guarantee at least 2; cap at max_m

    def find_top_k_references(X, x_test, k):
        score every reference by ctx_sim
        return top k indices, ctx-distances, raw feature distances
```

### 25.3 Implement the agent

```
class MolActAgent:
    def fit(X_ref, y_ref, task=None):
        detect task if not given (numeric+>10-distinct-or-non-int
            -> regression; otherwise classification)
        coerce y_ref dtype accordingly
        run variance filter (drop constant columns)
        run inner _validate (shapes, dtypes, regression-specific
            checks: ≥3 distinct, non-degenerate range, no NaN)
        scale features: StandardScaler/MinMaxScaler/none
        store X_ref_scaled, X_ref (raw), y_ref
        if classification: class_labels = sorted unique
        if regression: target_range = (min, max)
        fit NeighborFinder on X_ref_scaled
        if regression: build CCR with two inner MolActAgent classifiers

    def predict_one(x_test):
        if regression: delegate to CCR.predict_one
        else: classification path

    def _predict_one_classification(x_test):
        scale x_test
        nbr = neighbour_finder.find_local_references(...)
        cands = neighbour_finder.select_top_m_classes(nbr)
        if < 2 candidates: return _empty_result(reason)
        all_combos = cartesian product of per-class indices
        drop combos where any reference equals the test sample
            within atol=1e-6 (degenerate)
        if no usable combo: return _empty_result
        for each combo in parallel:
            res = _classify_tuple(refs, x_test, combo)
        result = _aggregate_results(tuple_results, ...)
        attach reference_info, test_feature_values
        return result

    def _classify_tuple(refs, x_test, ref_indices):
        compute discriminative mask (features not all equal across
            the M references)
        if no discriminative features:
            return Unknown short-circuit record
        compute n_aligned[c] and n_neutral over discriminative cols
        baseline_class = argmax(n_aligned), Unknown on tie or all-zero
        for each candidate-class reference:
            attach its k_local neighbourhood block (indices,
                their class labels / target values, their features
                projected to the discriminative subset)
        msgs = build_tuple_prompt(refs[disc], class_names, x_test[disc],
                                  feature_names[disc],
                                  n_total_features=full_count,
                                  n_aligned, n_neutral, baseline_class,
                                  reference_neighbourhoods)
        for attempt in 0..max_check_retries:
            raw = llm.chat(msgs)
            parsed = safe_parse_tuple_json(raw)
            ok, fixed, issues = _check_tuple(parsed, ...)
            if ok: return _finalize_tuple(fixed, ...)
            msgs = build_feedback_messages(msgs, raw, issues)
        return deterministic-fallback record:
            label = baseline_class
            class_scores = n_aligned/total
            confidence = 0.5 * margin + 0.4

    def _finalize_tuple(parsed, ...):
        compute class_scores from n_aligned (NOT from LLM strengths)
        clamp label to baseline_class if it disagrees AND both are
            real candidate classes
        return canonical tuple record (see §16.6)

    def _aggregate_results(tuples, ...):
        class_votes[c]   = count of tuples where label==c
        class_weights[c] = sum of confidence over those tuples
        if no decisive tuple: final = Inconclusive
        else:
            sort by weight descending
            threshold = 0.05 * max(n_decisive, 1)
            if winner > runner + threshold: final = winner
            else: final = Inconclusive
            margin = (winner_w - runner_w) / max(winner_w + runner_w, eps)
        compute confidence per §18.2 formula
        compute driver_attribution per §18.3
        return aggregated dict (see §18.5)
```

### 25.4 Implement CCR

```
def equal_frequency_bins(y, n_bins):
    use pd.qcut with duplicates='drop'
    fall back to np.linspace + np.digitize on failure
    fall back to a single bin as last resort
    return bin labels "bin_0".."bin_{N-1}" and edges

def cosine_feature_clusters(X, n_clusters):
    Xs = StandardScaler().fit_transform(X)
    cos = pairwise_distances(Xs, metric='cosine')
    cluster with AgglomerativeClustering(metric='precomputed',
        linkage='average', n_clusters=n_clusters_eff)
    return labels "fc_0".."fc_{M-1}"

def resolve_n_classes(n, mode='sqrt'):
    upper = max(2, n - 1)
    if mode='sqrt': return clamp(floor(sqrt(n))+1, 2, upper)
    elif mode='sqrt-floor': return clamp(floor(sqrt(n)), 2, upper)
    elif mode='sqrt-capped-10': return clamp(floor(sqrt(n)), 2,
        min(10, upper))
    elif mode='fixed': return clamp(fixed_value, 2, upper)

class CCR:
    def fit(X, y):
        N = resolve_n_classes(len(X)) (or override)
        M = resolve_n_classes(len(X)) (or override)
        bins, edges = equal_frequency_bins(y, N)
        clusters    = cosine_feature_clusters(X, M)
        agent_target  = MolActAgent(...).fit(X, bins, task='classification')
        agent_feature = MolActAgent(...).fit(X, clusters,
                                             task='classification')

    def predict_one(x):
        result_t = agent_target.predict_one(x)
        winner_t = result_t.final_label OR weighted-vote leader OR None
        nn_t     = top k_per_class refs in winner_t by ctx-distance
                   (returns []  when winner_t is None)
        # same for Phase 2 / feature view
        merged   = dedupe by ref index keeping smallest distance
        if empty: return refusal record

        if aggregator in {'conf', 'conf-distance'}:
            for each phase (target, feature):
                for each (ref_idx, dist) in that phase's neighbours:
                    weights[ref_idx] += phase_conf  (or phase_conf/dist)
            return sum(target[r]*w[r])/sum(w[r])
        elif aggregator='median': median of selected targets
        elif aggregator='weighted': 1/(d+eps) weighted mean
        else (mean): uniform mean of selected targets

        confidence = mean of phase confs that produced a winner
        interval   = (min(targets), max(targets))
        attribution= synthesize per-feature with Pearson direction
        return result dict (see §19.9)
```

### 25.5 Implement the prompts

The prompt strings are reproduced verbatim in
[`molactagent/prompts.py`](../molactagent/prompts.py). The
structurally important pieces that any reimplementation must
preserve:

- The discriminative-feature filter note ("Showing N
  discriminative features out of T total").
- The DETERMINISTIC ALIGNMENT TALLY block, surfaced as ground
  truth, listing per-class counts and the baseline verdict.
- The PER-REFERENCE LOCAL CONTEXT blocks, one per reference.
- The JSON-only output requirement, with the exact schema:
  `label`, `confidence`, `key_drivers` (3-6 entries),
  `supporting_features`, `opposing_features`, `summary`,
  `counter_evidence`, `uncertainty`.
- The "default to baseline unless decisively contradicted" rule.

### 25.6 Implement the LLM client

A minimal wrapper would:

- Accept `messages`, `temperature`, `tag`, `max_tokens`.
- Loop up to `max_retries`, sleeping
  `retry_delay * (attempt+1)` seconds (doubled when the error
  reads as a 429/throttle/rate-limit).
- Track `_call_count`, `_total_prompt_tokens`,
  `_total_completion_tokens`, `_total_time`, `_throttle_count`
  under a thread-safe lock.
- Provide a `ping(timeout)` method that bypasses the retry loop
  and classifies failures as `auth`, `model`, `network`, or
  `unknown`.

### 25.7 Implement validation, IO, pipeline, and reports

- `validate_inputs` collects every error/warning before returning;
  see §12.2 for the full check list.
- `prepare_for_agent` does the minimum coercion and returns
  numpy arrays plus metadata.
- `run_prediction` chains validation → preparation → LLM build
  → preflight ping → agent fit → per-sample loop → aggregate
  artefacts. Failure modes are described in §21.2.
- Reports are pure renderers over the result dicts; the schemas
  in §16.6, §18.5, and §19.9 are everything they need.

### 25.8 The behavioural acceptance test

A reimplementation passes if, on the bundled iris example:

- `setosa` is excluded by the adaptive top-M selector for every
  test sample (it sits far enough from versicolor/virginica).
- M=2 contrasts dominate; the central setosa-versicolor or
  versicolor-virginica pairs are picked.
- Boundary samples between versicolor and virginica come back
  with substantially lower confidence than samples deep in either
  cluster.

And on the bundled penguins example:

- The CCR LOOCV mean absolute error sits within ~5% of 354 g and
  the median absolute error is the lowest of all baselines (KNN3,
  Ridge, Linear, Random Forest).

These match the reference numbers in §27.

## 26. What good behaviour looks like

A few invariants worth building tests around:

- **Deterministic across reorderings of tuples.** Result records
  must be returned in input-position order so aggregation is
  reproducible regardless of which tuple finishes first.
- **`class_scores` come from the alignment tally.** If you
  compute them from LLM strengths, you have re-introduced the
  bias the design rejected in §5.3.
- **Per-tuple `label` clamps to `baseline_class`.** A passing
  test should set up a scenario where the LLM and tally
  disagree, and assert the final tuple label equals the
  baseline.
- **Neutral features count separately.** A tuple where all
  discriminative features tie should yield `n_aligned = 0,
  n_neutral = n_disc`, a baseline of `Unknown`, and the LLM's
  verdict (if any) should not be clamped.
- **Confidence is monotone in margin.** Holding sample size and
  coverage constant, larger margins must produce larger final
  confidence values.
- **CCR refuses cleanly.** If both phases return `Unknown`
  (e.g. because all reference distances tie), `predicted_value`
  must be `None` and the result must still pass through
  pipeline / report rendering without exceptions.

The bundled test suite covers all of the above plus extensive
schema, contextual-similarity, prompt-structure, and
report-rendering coverage. It runs in under 10 seconds against a
mock LLM (`tests/conftest.py::FakeLLM`) and requires no API
access.

---

# Part V — Empirical Foundations

## 27. Benchmarks

### 27.1 Penguins LOOCV — n = 24, body mass in grams

Leave-one-out cross-validation against the same reference set
used for the classical ML baselines (the small subsample built
by `scripts/prepare_penguins_subsample.py`):

| Method | MAE | R² | Pearson r | Median AE |
|---|---:|---:|---:|---:|
| KNN3              | 369.10 | 0.523 | 0.736 | 300.00 |
| Ridge             | 355.17 | 0.542 | 0.738 | 311.76 |
| Linear regression | 350.36 | 0.539 | 0.739 | 257.71 |
| Random forest     | 360.01 | 0.540 | 0.746 | 334.50 |
| **MolActAgent (CCR)** | **353.89** | **0.532** | **0.755** | **242.70** |

MolActAgent matches the best classical baselines on MAE / R² /
Pearson r, and holds the lowest median absolute error of any
method tested — the most *typically* accurate regressor in the
panel. (Median AE is robust to the same outliers that distort
mean AE, so a low median says "most predictions are tight" even
when mean AE looks middling.)

### 27.2 antiTB binary classification LOOCV

Reproducing the pre-refactor binary baseline:

- Balanced accuracy: **0.833**
- Matthews correlation coefficient: **0.632**

Bit-identical to the original binary pipeline — the multi-class
rewrite did not regress the M=2 special case. See
[`data/refactor_validation.md`](../data/refactor_validation.md)
for the full numbers and source files.

### 27.3 The validation philosophy

Both benchmarks used leave-one-out cross-validation explicitly
(rather than k-fold) because the small reference sets of interest
do not give enough rows to populate larger folds without losing
class diversity. LOOCV is also the cleanest way to expose memory
artefacts — every test sample is *out* of the reference set the
classifier sees, so the system cannot accidentally cheat by
looking up its own answer.

The protocol scripts live in `scripts/run_loocv_metrics.py`,
`scripts/run_loocv_BSLT_regression.py`, and
`scripts/aggregate_loocv_results.py`. Re-running them against a
real LLM endpoint requires only an `llm_config.jsonc` and the
respective input dataset.

## 28. Known limitations and open questions

These are unresolved and worth tracking. Each is a place where
future implementation work will need to make a concrete choice
that this document deliberately leaves open.

1. **Empirical reliability of LLM landscape-reading for
   regression on larger datasets.** The penguins n=24 result is
   encouraging but the design's regression branch has not been
   stress-tested at, say, n=300 with a target like log-IC50 in a
   real bioactivity setting.

2. **Genre cues in the prompt.** Even though the mechanism is
   unified across classification and regression, the LLM's
   default behaviour differs between "predict a class" and
   "predict a number." We may need to tell it explicitly which
   task it is doing — though for CCR this is moot because both
   phases are *literally* classifications.

3. **Whether the test sample's own neighbourhood should be
   shown.** Each reference brings its neighbours. The test
   sample also has a neighbourhood (computed against the
   reference set). Showing it adds a vantage point; it may also
   be redundant with what the references' neighbourhoods convey.
   Currently not shown.

4. **Choice of similarity blend.** The 50/50 default is an
   initial guess. The right value probably depends on the
   intrinsic dimensionality and density of the reference set; it
   may need to be tuned per dataset.

5. **How asymmetric texture is presented for classification.**
   The current per-reference-neighbourhood blocks let the LLM
   perceive boundary vs. deep-territory references implicitly.
   Whether some explicit Python summary (a "purity score") would
   help on harder problems is open. The default position is to
   *not* compute one, per §7.2.

6. **Prompt size and cost.** The texture blocks multiply the
   per-tuple prompt size by roughly k_local. Current per-tuple
   prompts run ~3000-10000 tokens; the cost is not prohibitive
   at current LLM pricing but is not free either.

7. **Concrete formula for neighbourhood similarity.** §6 leaves
   the choice open; the implementation chose mean min-distance
   cross-coverage (§14.1) for its simplicity. A different formula
   (bipartite matching, EMD) might give marginally different
   rankings; benchmark first before changing the default.

8. **Per-phase aggregator weighting.** CCR's default is
   confidence-weighted (each picked reference weighted by its
   source phase's classification confidence; references picked
   by both phases get summed weights). Whether
   `"conf-distance"` (further weighted by `1/(d+eps)`) helps on
   datasets with sparse reference coverage near boundaries is an
   empirical question the suite has not pinned down.

## 29. Status and extensibility guidance

As of this writing, MolActAgent implements every principle laid
out in Part II. The system is task-agnostic at every layer; the
same filter / evidence-assembler / reasoner pipeline serves both
classification and regression.

### 29.1 Implemented features (and where they live)

| Feature | Module(s) |
|---|---|
| Multi-class adaptive top-M classification | `molactagent/agent.py`, `molactagent/neighbors.py` |
| Contextual similarity (feature ⊕ neighbourhood) | `molactagent/neighbors.py` |
| Dynamic per-reference texture blocks | `molactagent/agent.py`, `molactagent/prompts.py` |
| Unified classification + regression flow | `molactagent/agent.py`, `molactagent/cross_classification.py` |
| CCR regression backend (target-bin × feature-cluster) | `molactagent/cross_classification.py` |
| Layered epistemic architecture (witness → assembler → reasoner) | system-wide |
| Soft baseline override + retry + final clamp | `molactagent/agent.py` |
| Rule-based JSON validation | `molactagent/agent.py` |
| Provider-agnostic LLM wrapper | `molactagent/llm.py` |
| Multi-format data loading and validation | `molactagent/io.py` |
| LLM-config resolution + multi-profile support | `molactagent/config.py` |
| Per-sample HTML / Markdown / JSON reports | `molactagent/report.py`, `molactagent/prompts.py` |
| Aggregate interpretability artefacts (zero extra LLM calls) | `molactagent/report.py` |
| Pipe-friendly CLI with stdin / JSONL / TSV / CSV output | `molactagent/cli.py`, `molactagent/io.py` |

### 29.2 Patterns to keep when extending

- **New numerical features belong in the witness layer.** If you
  add a new similarity / weight / score, surface it to the LLM as
  ground truth in the prompt; do not let it silently override
  the reasoner.
- **New qualitative features belong in the prompt assembler.**
  If the LLM needs to perceive a new pattern, give it the
  underlying data and let it perceive it. Do not pre-summarise
  if you can avoid it.
- **The class-label space is opaque.** Class labels flow through
  the system as user-supplied strings. Avoid re-introducing a
  "Pos / Neg" or "positive class" distinction; the multi-class
  pipeline treats every class the same.
- **Schema additions, not replacements.** Both per-tuple and
  aggregated result schemas are consumed by reports and tests;
  new fields are fine, removing fields is not without notice.
- **Test against the FakeLLM first.** `tests/conftest.py::FakeLLM`
  emits a syntactically valid response that satisfies the
  baseline-clamp policy. Most behavioural changes are testable
  against it without ever touching a real API.

### 29.3 Extension dos and don'ts

- **Do** add new aggregators (CCR has `conf`, `conf-distance`,
  `mean`, `median`, `weighted`; adding e.g. a trimmed-mean is
  straightforward inside `_aggregate_per_phase`).
- **Do** add new LLM provider templates to the starter JSONC.
- **Do** add new pre-flight diagnostics to `_preflight_llm` if
  you can classify a failure mode cleanly into one of `auth`,
  `model`, `network`, `unknown` (or add a new kind, with care).
- **Don't** introduce a per-feature continuous "weight" the
  Python layer uses to score classes. The deterministic
  alignment tally is the only weighting Python is allowed to
  perform on per-feature evidence.
- **Don't** add a hidden "high-confidence override" path that
  silently bypasses the soft baseline check. If you want to
  loosen the override, change the `+ 1.0` threshold and put a
  test on the new value — do not fork the policy.
- **Don't** push prompt-construction logic into `agent.py`; new
  prompt machinery belongs in `prompts.py` so the prompt surface
  area stays inspectable from one file.

If you find yourself making a change that this document does not
cover, or that contradicts it, the right move is to revise this
document first and then make the code change — not the other way
around. The code is allowed to outrun the document only on
details that are genuinely too small to matter; principle-level
changes belong here first.
