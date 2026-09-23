# Analysis index and documentation checks

This index was checked against source code and the September 22, 2026 manuscript and supplementary material. It identifies available source, without treating an entire directory as proof that every panel can be reproduced.

## Available source

| Purpose | Source | Scope |
| --- | --- | --- |
| Main mediation and theta-control models | [Closed-loop notebook](../analysis/mediation/closed_loop_mediation.ipynb) | Figure 5e–g analyses; additional all-region model labeled exploratory |
| ESR1 mediation | [ESR1 notebook](../analysis/mediation/esr1_mediation.ipynb) | Supplementary Figure S8 analysis variants; implementation notes in its README |
| NMF model implementations | [models/NMF](../analysis/network/models/NMF) | Historical TensorFlow implementations and variants |
| Single-region and two-region behavioral prediction | [single_region](../analysis/regional/single_region), [two_region](../analysis/regional/two_region) | Separate task-specific models and comparisons; identify the selected experiment before interpreting results |
| Latent-dimension exploration | [BIC_Model.ipynb](../analysis/regional/component_selection/BIC_Model.ipynb), [Reconstructions.ipynb](../analysis/regional/component_selection/Reconstructions.ipynb) | Generative reconstruction/BIC-like calculation; not by itself a verified final supervised-model selection pipeline |
| Reduced encoder fitting | [Cross_validateEncoder.ipynb](../analysis/network/experiments/CombinedDatasets/Cross_validateEncoder.ipynb) | Power/coherence ElasticNet approximation with grid search and a subsequent fixed-parameter fit |
| Offline threshold exploration | [thresholds](../analysis/network/thresholds) | ROC/precision-recall exploration; not the live stimulation controller or a complete final threshold export |
| Network-weight uncertainty exploration | [ConfidenceIntervals.ipynb](../analysis/network/experiments/ConfidenceIntervals.ipynb), [PlotValuesBootstrap.ipynb](../analysis/network/experiments/PlotValuesBootstrap.ipynb) | Contains exploratory variants; not certified as the manuscript's final 1,000 animal-bootstrap procedure |

## Concrete limitations retained from the source

- `analysis/network/experiments/runElastic.py` is **not a working training entry point as saved**: it requires a command-line argument and its fitting/saving block is inside a triple-quoted string. Its `nFact = 8` value does not make the block executable. Other model-training variants are retained for author selection.
- `analysis/network/experiments/Causality/` and `Causality2/` are historical predecessors. In their `Causality_Analysis.ipynb`, the full-model cell constructs `X2_2` but fits `X1_2`. Use `analysis/mediation/closed_loop_mediation.ipynb` for the mediation analysis, rather than those predecessors.
- Local paths such as `/home/austin/...` and `/media/austin/...` remain in historical source. External helper modules, precomputed feature files, and fitted models must be supplied/configured before execution. In particular, `utils_np` is imported by several notebooks but is not supplied here.
- The reduced-encoder notebook's grid-search estimator is followed by a fixed ElasticNet parameter choice. It does not by itself establish which export was deployed in the experiments.
- `ConfidenceIntervals.ipynb` includes a row-bootstrap variant and an animal-label sampling variant with a default of 20 repeats. The latter selects rows using `np.isin`, which does not preserve multiplicity when an animal is sampled more than once. Do not describe this notebook as verified reproduction of the draft's 1,000 animal-bootstrap calculation without identifying/correcting the final procedure.
- The mediation companion README distinguishes the implemented ESR1 mouse-indicator adjustment and centered smoothing from the supplement's mixed-model/trailing-average wording.

These are documented source limitations, not changes to the computational analyses. No new p-values or results were generated for this migration.

## Claims removed or narrowed from the upstream README

- Removed the assertion that the repository contains everything needed to reproduce all findings, and the unsupported directory-to-figure mappings for cellular coupling, LinCx experiments, locomotor tracking, and stimulation hardware.
- Replaced the earlier mediation link with the actual mediation notebooks.
- Removed runnable-training instructions for a disabled entry point and claims that its filename proves a 12-factor model. The manuscript reports eight total networks.
- Replaced unverified broad dependency/version guidance with a scoped mediation setup and the manuscript's historical modeling environment.
- Linked LFP preprocessing to the existing shared repository rather than presenting it as a missing deliverable.
- Removed general biological exposition, patent statements, and bibliographic claims that are unnecessary to explain the code. The main manuscript title and author list are transcribed from the supplied draft; no journal DOI or software DOI is invented.
- Narrowed overstatements in three latent-dimension notebook narratives: cosine similarity is not a significance test, and the manuscript reports a three-network exception to encoder similarity.

## Naming and provenance

The [source manifest](source_manifest.json) records original and new paths, source hashes, and omitted upstream files. Only source files were imported from Austin's collection. Distinct source found only inside ZIP archives is extracted under `analysis/network/archive_sources`; archives and their data payloads are excluded. Notebooks with identical code-cell text are consolidated into one copy; the manifest maps each removed duplicate to its retained notebook. Distinct code variants remain available under their existing filenames.
