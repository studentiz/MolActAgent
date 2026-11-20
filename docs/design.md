# How MolActAgent makes a prediction

Can the measured reference compounds support an activity assessment for
your candidate in this assay? Compare the candidate with compounds tested
under comparable conditions. The recorded decision shows which references
supported a class and where the available evidence left the question open.

MolActAgent has two engines with different decision rules. The SMILES
engine asks a language model (LLM) to judge a panel of active and inactive
compounds. The feature-table engine compares rows of numeric features and
enforces a numeric baseline for conflicting class labels within each
comparison. Both engines address the assay and class definitions you
supply.

- [Molecular binary classification](#molecular-binary-classification) explains assay inputs, reference panels and the recorded CDK2 example.
- [Focused molecular comparisons](#focused-molecular-comparisons) explains when a local comparison can replace the panel result.
- [Feature-table engine](#feature-table-engine) explains numeric comparisons, tuple labels and weighted class support.
- [Model service and run checks](#model-service-and-run-checks) covers live configuration, validation and offline replay.
- [Limits of the evidence](#limits-of-the-evidence) describes how to interpret a prediction at the bench.

## Molecular binary classification

Use `molactagent molecular` or `molactagent.molecular.MolecularClassifier`
with reference and query structures written as SMILES. Both interfaces
use the same classifier.
The reference table supplies the `active` and `inactive` classes. The
query supplies its structure. The assay description tells the model what
those classes mean and which experimental question to answer.
[SMILES tutorial](tutorial.md#predict-from-smiles) · [Classifier source](../molactagent/molecular.py)

### Supply comparable measurements

Use reference labels from a common endpoint and comparable assay conditions.
For example, two kinase assays can report IC₅₀ in nM and still differ in
ATP concentration, enzyme construct or readout. Such differences can
change the apparent potency, so the assay identity and protocol matter
when choosing analogs. Record the conditions, the units, the meaning of
each label and how you handled replicates or qualified values such as
`>10 µM`. The classifier checks the inputs it receives. Assay curation
remains the user's responsibility.

Use stable IDs and specify stereochemistry in the SMILES where known.
RDKit parses the structures and produces canonical isomeric SMILES.
MolActAgent rejects invalid structures and conflicting labels for an
identical canonical reference structure. For each query it excludes
references with the same canonical SMILES. The query is `Inconclusive`
if this leaves either class absent.

Optional reference pChEMBL values supply numeric activity evidence to the
model. For an exact IC₅₀ measurement, pChEMBL expresses the negative
base-10 logarithm of its value in mol/L. Pass the numeric column with
`--pchembl-column`. The optional `measurement` text with the original
value and units and the HTTP(S) `source_url` are display fields for the
reference records.

`--pchembl-threshold` alone gives the model assay context. Add
`--labels-from-pchembl` to check every supplied reference label against
one cutoff. In that mode every reference needs a finite pChEMBL value,
and `active` means at or above the cutoff. For an assay with separate
active and inactive limits, supply labels and state both limits in the
assay description.

### Retrieve analogs and build the panel

1. **Search within each class.** Rank by Tanimoto similarity on RDKit's
   2,048-bit, radius-2 Morgan fingerprints. Tanimoto measures the overlap
   of fingerprint bits. Searching both classes gives the model a
   contrast even when most close references share one label. The default
   maximum is eight per class, reduced to four at that setting when six
   or fewer unique reference structures remain. A class with fewer
   eligible rows contributes its available rows.
2. **Give the model structural context.** The prompt includes the
   endpoint, canonical SMILES, reference labels, supplied pChEMBL,
   fingerprint similarities and computed structural features. These
   include ring and functional-group descriptions, stereochemistry and
   descriptors such as molecular weight and polar surface area. The
   prompt can add related rows for same-scaffold, constitutional-isomer,
   halogen or stereochemical comparisons. It also summarizes some
   fingerprint features that differ in prevalence between classes.
3. **Ask the chemical question.** The model is asked to name a structural
   contrast, place the query in relation to it, cite supporting reference
   rows and describe the strongest counter-evidence or missing
   comparison. It may return `Uncertain`. The reported main panel is the
   nearest set from step 1. The full evidence references include the
   additional rows actually available for citation.

The same scaffold can accommodate changes in substitution or
stereochemistry that produce large changes in activity. A high
fingerprint similarity helps find analogs for closer inspection. An
activity-cliff claim needs measured compounds that isolate the proposed
change. [Evidence extraction and prompts](../molactagent/_molecular_evidence.py)

### Check the replies and decide

By default MolActAgent requests **five judgments on the same panel**. A
valid reply supplies an allowed class or `Uncertain`, a finite numeric
score from 0 to 1, a written rationale and counter-evidence. Citation IDs
must refer to rows available in that comparison. A class judgment needs
at least one supporting reference. This checks response structure and
citation membership. Chemists can inspect the cited structures and
measurements to judge whether they support the argument.

The engine can ask once for a correction after a truncated, invalid or
abnormally finished reply. The model client can also retry transient HTTP
errors. Both kinds of retry belong to the original planned judgment.
Valid `Uncertain` replies count as abstentions. A judgment that ends in
request or response failure is counted separately and casts no vote.

| Decision step | Rule |
| --- | --- |
| Require usable judgments | More than half of the planned panel judgments must give `active` or `inactive`. With five calls planned, at least three must give a class. Otherwise the result is `Inconclusive`. |
| Compare the class counts | The larger count of valid class judgments wins. The final score is the average model score of the winning judgments. |
| Resolve equal counts | The larger sum of class judgment scores wins unless the two sums are effectively equal. The saved score is the winning sum divided by the combined sum. Equal counts and effectively equal sums give `Inconclusive`. |

There is no minimum model score for a valid class vote in the main panel.
The evidence score summarizes the returned judgments. Repeated calls see
the same evidence and can give correlated responses. A score of 0.8
gives no estimated 80% chance of the class being correct, and the engine
assigns no numeric potency to Q1.

### Worked example from the CDK2 assay

The v0.5.1 case uses ChEMBL assay **CHEMBL897015**, a biochemical assay of
human cyclin-dependent kinase 2 (CDK2) inhibition from Lee and colleagues'
published study of 3-hydroxychromones. The case defines active references
as IC₅₀ ≤ 100 nM (pChEMBL ≥ 7) and inactive references as IC₅₀ ≥ 10,000 nM
(pChEMBL ≤ 5). Intermediate measurements are outside these two example
classes. The cutoffs describe this example's label selection.
[ChEMBL assay](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL897015) · [Original study](https://doi.org/10.1016/j.bmcl.2006.12.011)

**Q1 is CHEMBL392776.** The run supplies its structure and withholds its
measured activity and class. Its closest active reference, CHEMBL232557,
has a phenolic hydroxyl at the site where Q1 has bromine. The rest of the
structure matches. The reference's measured **87 nM** is close to the
100 nM active cutoff. All four references are halogen-free. The closest
inactive reference, CHEMBL50 (**40,000 nM**), differs at several sites,
so its measurement cannot isolate the effect of the hydroxyl-to-bromine
substitution.

The recorded result was **Inconclusive**. One judgment favored active
and four abstained because the effect of the substitution remained open.
At least three of the five planned judgments had to supply a class.
All five returned valid replies, with zero failed calls.
[CDK2 input files](../examples/cdk2/) · [Open the CDK2 report](https://studentiz.github.io/MolActAgent/cdk2-report.html)

### Focused molecular comparisons

The default `panel` profile stops at the panel decision. In `focused`
(`--molecular-profile focused`), the engine may make additional local
comparisons once the main panel meets its required count of decisive
judgments, including when those judgments leave a tied panel.

| View | When it runs and how it can decide |
| --- | --- |
| Halogen deletion | The query must have a scaffold and fewer F, Cl, Br and I atoms in total than its closest same-scaffold reference. A valid class judgment can replace the current result at a score of at least **0.50** if it differs from the current label, or **0.60** if it agrees. The trigger uses a halogen count. The model is asked to examine the site of substitution and whether available references isolate the change. |
| Stereochemical comparison | The query and an active reference must have identical connectivity after removing stereochemistry, preserving isotope identity. Their canonical structures must differ and both must have fully specified stereochemistry. The comparison uses this active reference and an inactive analog selected by scaffold match, terminal-group similarity and fingerprint similarity. A valid class judgment with score **≥ 0.60** is used if the paired comparisons can be formed. |
| Stereochemical paired fallback | The engine also requests judgments on up to three active–inactive pairs for the stereochemical view. If the direct stereo judgment is ineligible, more than half of the planned pair judgments must give a class and the score-weighted class totals must have a clear winner. The selected class score is its share of the combined pair weights. |

The halogen result is considered first. An eligible stereo result can
supersede it. A selected focused result supplies the final label and
score, even when it agrees with the panel. The saved decision records the
selected view, the original panel counts and which focused decisions were
applied. These rules operate on model judgments. Assigned R/S labels and
halogen counts alone provide no activity rule.

### The hydroxyl teaching case

The secondary example uses a synthetic rule. A compound is active if its
hydroxyl is bonded directly to an aromatic carbon. 4-Fluorophenol and
p-cresol share this attachment. Benzyl alcohol has a methylene group
between its hydroxyl and its aromatic ring. The simple contrast makes it
easy to follow the reference citations and see how the model describes a
missing comparable substitution. [Open the hydroxyl report](https://studentiz.github.io/MolActAgent/molecular-example.html)

## Feature-table engine

Use `molactagent run` or `MolActAgentClassifier` when you already have one
row of numeric features per compound. These could be molecular
descriptors or experimentally measured properties with consistent units.
The engine also handles general numeric tables and more than two classes.
It works from the features and class labels you supply. If two structures
have the same descriptor vector, those features cannot distinguish
them. Give features meaningful names and include their units, definitions
and assay background so the model can interpret the columns.
[Feature-table technical reference](feature-table-design.md) · [Numeric-table tutorial](tutorial.md#iris-tutorial-for-numeric-tables)

### Find local comparisons

- **Prepare the features.** The CLI matches the tables by column name and
  sets aside the label and any designated ID column. The engine removes
  zero-variance reference columns by default. It fits scaling on the
  reference data and applies it to the query for retrieval. Standard
  scaling is the default, with min–max and no scaling also available.
  By default the tuple prompt uses the retained features in their original
  units.
- **Rank by feature and neighborhood similarity.** The default combines
  direct feature similarity and similarity between the two rows' local
  reference neighborhoods in equal proportions. The neighborhoods use
  up to three nearest references by default. This lets nearby examples
  with similar surroundings influence retrieval. The prompt also shows
  each chosen reference's neighboring rows and labels, which can reveal
  mixed labels near a class boundary.
- **Keep competing classes.** The CLI and pipeline default to up to three
  neighbors per class. With more than two classes, the engine considers
  how close each class's closest reference is. By default it admits
  distances up to 1.5 times the smallest contextual distance, with a
  small floor for a near-zero distance. It uses the closest two classes
  if fewer qualify and the default cap is four. A binary task uses both
  classes when they are available.
- **Form tuples.** Each tuple consists of the query and one reference from
  each candidate class. The engine tries each combination of the chosen
  references and skips tuples with a reference whose feature vector is
  nearly identical to the query. The number of combinations grows as the
  product of the class pool sizes, and tuples can share references.

For implementation details, feature similarity is `max(0, 1 − d/D)`,
where `d` is the feature-space distance and `D` is the largest fitted
reference-pair distance with a floor for zero. Neighborhood comparison
uses the same normalization. Its distance is the average of the mean
nearest distances from each neighborhood to the other. Contextual
distance is one minus the blended similarity. Two compounds can differ
on individual descriptors while sharing many of the same neighboring
references. [Retrieval source](../molactagent/neighbors.py) · [Preprocessor](../molactagent/preprocessing.py)

### Decide a tuple from its feature alignment

For each tuple, keep features whose values differ across the references.
On each retained feature, the class of the uniquely closest reference
value gets one alignment count. A tie is neutral. For example, if the
query has 1 donor, and the two references have 0 and 2, that column adds
no count. The unique class with the largest positive tally is the
numeric baseline. A tied or zero tally gives `Unknown`. A tuple with
identical reference values on every feature returns `Unknown` without
asking the model.

The model sees that tally alongside the feature values and neighboring
rows. It proposes a class or `Unknown`, reports a score and identifies
drivers, counter-evidence and uncertainty. The response check requires a
recognized label and at least two valid drivers citing distinct retained
features and candidate classes. It normalizes numeric scores to the
allowed range and can ask for corrections.

| Situation | Saved tuple label |
| --- | --- |
| The baseline is a candidate class and the model proposes a different candidate class | The engine saves the numeric baseline. If the model's driver strengths for its proposed class sum to more than the strengths for the baseline class plus **1.0**, the reply can pass without a correction on that point. The final label still follows the baseline. |
| The model returns a valid `Unknown` | The tuple stays `Unknown` and casts no class vote, even if the baseline names a class. |
| The baseline is `Unknown` and the model supplies a valid candidate class | The model's class can become the tuple label. |
| Response validation runs out of attempts | The engine uses the numeric baseline, with an alignment-derived weight and an explanation of the fallback. It keeps no model drivers for that tuple. |
| The model call fails with an exception | The engine records an `Unknown` tuple with the call error. |

The tuple's `class_scores` come from normalized alignment counts, or
uniform values when all counts are zero. A saved model rationale can
favor another class after a conflicting label has been resolved by the
baseline. Read the saved `label` and `baseline_class` alongside the
interpretation. The validator's driver-strength check is a rule for
requesting a correction, with up to three correction attempts by default.

### Combine tuple results

A decisive tuple adds its weight to the total for its **saved** class.
The aggregation uses the accepted model score, or a weight of 0.5 for a
missing or zero score. For a validation fallback, the weight is
`0.4 + 0.5 × (highest alignment proportion − runner-up proportion)`.
`Unknown` tuples add no class weight.

The leading class must exceed the runner-up's summed weight by **more
than 0.05 × the number of decisive tuples**. If the lead falls short,
including an exact tie, or if no tuple is decisive, the result is
`Inconclusive`. Model scores can therefore affect the overall winner even
where numeric alignment controls individual tuple labels.

The final `confidence` is a heuristic. With a winning class, it
multiplies `0.5 + 0.5 × relative weight margin` by a factor that rises
with reference size and coverage. The relative margin is the winning
weight minus the runner-up weight, divided by their sum. The multiplier is
`0.5 + 0.25 × reference size score + 0.25 × coverage`. The reference size
score is the smallest candidate class's full reference count divided by
ten, capped at one. Coverage is the fraction of evaluated tuples that
gave a class. An inconclusive aggregation normally displays 0.4. An empty
result from before aggregation displays 0.

The estimator's `predict_proba` exposes normalized class support from
the weights. Its `predict_with_confidence` returns the label with the
separate final evidence score. Both values summarize the run and provide
no estimate of accuracy on new compounds. Driver attributions count
feature citations across tuples and sum the reported strengths by
favored class. They help you trace the model's interpretation back to
individual comparisons.
[Estimator interface](../molactagent/estimators/classifier.py) · [Numeric report guide](reports.md#reports-from-numeric-tables)

<a name="19-the-ccr-regression-backend"></a>

### Experimental regression

The feature-table path also offers experimental cross-classification
regression (CCR), exposed by `MolActAgentRegressor`, `run_regression` and
the numeric-target CLI. It groups the reference rows in two ways.
One view uses roughly equal-frequency target bins. The other clusters
the standardized feature vectors by cosine distance with average
linkage. The default requested group count is based on
`floor(sqrt(n)) + 1`, subject to the size and distinct values in the
reference set.

Each view uses the feature-table classifier. The regressor draws up to
two contextual neighbors from each selected class by default. A view
with an inconclusive classification can still supply a class from its
recorded class weights. CCR averages the selected reference targets,
weighted by each view's classification score by default. A reference
selected by both views receives weight from both. CCR can refuse when
the views yield no usable references or the aggregation has no usable
weight.

The interval spans the minimum and maximum selected reference targets.
It describes their spread and has no statistical coverage guarantee.
The confidence score is the mean classification score of the views that
supplied a class to CCR. The code remains experimental, and the
accompanying manuscript evaluates classification.
[CCR implementation reference](feature-table-design.md#experimental-cross-classification-regression) · [Regression status](experimental-regression.md)

## Model service and run checks

### Live model configuration

Both CLI engines use a service that implements the OpenAI-compatible
**Chat Completions** interface. To start, run `molactagent init-config` and
provide the service's `base_url`, `model_name` and `api_key`. Model names
and request options must be supported by the chosen endpoint. For
molecular runs the client requests JSON mode by default. If the service
can return JSON as text and rejects the JSON-mode parameter, use
`--plain-json`. Reply validation still applies.

For each field, explicit CLI options take precedence over supported
environment variables, which take precedence over the selected config
file. The connection variables are `MOLACTAGENT_API_KEY`,
`MOLACTAGENT_BASE_URL` and `MOLACTAGENT_MODEL`. The loader also reads
`MOLACTAGENT_TEMPERATURE` and `MOLACTAGENT_TIMEOUT`. Pass `--llm-config` to
select a file. Otherwise the loader looks upward from the current
directory for a known configuration filename, starting with
`llm_config.jsonc` at each level. If a file contains multiple profiles,
`--profile` selects one and the default selection is the last one.
`--molecular-profile` selects the chemical comparison mode.
[Live-model tutorial](tutorial.md#use-a-live-model) · [Configuration loader](../molactagent/config.py)

The molecular CLI uses the configured temperature and output budget,
which default to 0 and 16,384 tokens. The feature-table engine's
classification tuple call requests 2,000 output tokens. The shared
client defaults to a 120-second timeout per HTTP attempt and up to ten
additional attempts for transient failures. A transport retry and a
request to correct the response are separate operations. Check the
service's policies before sending structures and assay information.

### Validate before interpreting

For feature tables, `molactagent validate` checks input files without
contacting the model service. It checks the label or target column,
numeric feature compatibility, missing data, class coverage and possible
overlap in the tables. A live feature-table pipeline also checks service
reachability when the client supports it. Molecular runs parse and check
their structures and labels during input preparation. Their API failures
appear in the call records. These checks cover input and response
consistency. Comparable assay conditions and credible chemical arguments
require experimental judgment.
[Input checks](../molactagent/io.py) · [Numeric pipeline](../molactagent/pipeline.py)

### Replay molecular decisions

Save a live run with `--record`. The schema 2 recording stores effective
settings, software versions, hashes of inputs and requests, and responses
or request errors. CLI recordings also store file fingerprints and column
mappings. Replay with `--replay` and the original input files. It rebuilds
and checks the requests and recomputes the decisions offline.

Match the recorded RDKit version. v0.5.1 can replay compatible v0.5.0
schema 2 recordings and render them with the updated report. File,
settings and request checks still apply. Keep the historical inputs
byte-for-byte, including their original headers. Editing the CSVs to
add provenance columns changes their file fingerprints. For older schema 1
recordings, use v0.4.1 with its original options and dependencies.
[Replay walkthrough](tutorial.md#replay-a-complete-prediction) · [Replay source](../molactagent/_molecular_recording.py)

## Limits of the evidence

- **Reference coverage sets the scope.** A close analog with a measured
  result in the same assay can support a local comparison. A distant
  scaffold, missing substituent control or ambiguous stereochemistry
  weakens the inference. Retrieval can still supply the closest available
  references when all are distant.
- **Structural similarity is a calculated comparison.** Fingerprints and
  descriptors give a reproducible view of the supplied structures. They
  do not account for the full assay environment, a binding pose or every
  relevant protonation state. A small change can alter potency sharply.
- **A prediction can guide the next experiment.** Five replies from one
  panel are repeated assessments of shared evidence. The public CDK2
  example illustrates one assay decision with held-out activity. Broader
  accuracy requires evaluation on separate experimental data. To
  confirm a chemical inference, compare it with a measured result under
  the stated assay conditions.
- **Keep enough detail to audit the call.** Preserve the original
  structures, assay definitions, result JSON and any recording. Follow
  reference citations and check plausible opposing interpretations,
  abstentions and failures alongside the winning result.
  [Guide to reading reports](reports.md)
