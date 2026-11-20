# How MolActAgent makes a prediction

A chemist who has measured a handful of compounds in one assay can use those observations to choose the next candidate. MolActAgent combines local, example-based learning with a language model. Deterministic chemistry supplies comparable structures and values. The language model weighs what those comparisons mean for the stated endpoint. The agent retains the evidence, checks the response and combines usable judgments.

For example, Q2 in the cell assay has fluorine where measured active R2 has chlorine and measured inactive R5 has a hydroxyl group. Those two measurements come from the same cell assay. The model used the substitutions to support its prediction for Q2. The chemical inference, the observed labels, and the actual records are available side by side. [Cell-assay report](https://studentiz.github.io/MolActAgent/cell-assay.html#compound-2)

The SMILES engine compares groups of active and inactive structures. The numeric-table engine compares a candidate with one feature row per candidate class. Both retain the model's validated decision. Their retrieval, validation and aggregation rules differ.

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
references with the same canonical SMILES. Each other distinct isomeric structure occupies one reference slot. The first replicate with a supplied pChEMBL value represents the structure, or the first row if all lack one. Other measurements and their individual IDs remain on its record. The query is `Inconclusive`
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

The language model receives the tally alongside the feature table, neighboring rows, feature definitions and assay background. It judges which differences matter here. The validator checks the cited feature names, candidate classes, and quoted candidate and reference values against the displayed precision. A proposed class needs a driver supporting it. The local record retains the measured alignment and the accepted model decision separately.

| Outcome of a comparison | Effect on the decision |
| --- | --- |
| Valid model judgment, including a class different from the alignment leader | The accepted class and its model score go into the vote. The record saves `model_label`, `baseline_class` and `decision_source`. |
| Valid model abstention | The record saves `Unknown` with source `model_abstention`. |
| No valid reply after the correction attempts | The record saves `Unknown` with source `invalid_response` and the validation issue. |
| Request failure or indistinguishable reference feature values | The reason is retained. The comparison has no class vote. |

In v0.5.1 the numeric engine could replace a valid model label with the alignment leader. Use the original tag to reconstruct those historical runs. A saved historical field has the meaning assigned by its originating version.

### Combine tuple results

A decisive tuple adds its weight to the total for its **saved** class.
The aggregation uses the accepted model score. A score of zero adds zero
weight. New judgments require a valid score. Historical tuples that lack a
score use a weight of 0.5. `Unknown` tuples add no class weight.

The leading class must exceed the runner-up's summed weight by **more
than 0.05 × the number of decisive tuples**. If the lead falls short,
including an exact tie, or if no tuple is decisive, the result is
`Inconclusive`. The recorded class votes and their model weights determine the winner.

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
which default to 0 and 16,384 tokens. Numeric classification tuples honor the configured client budget. The shared
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

Match the recorded RDKit version. v0.6.0 selects the original panel instructions when replaying v0.5.0 or v0.5.1
schema 2 recordings and renders them with the updated report. File,
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
