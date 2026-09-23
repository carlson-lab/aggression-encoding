# Aggression encoding

Analysis code accompanying **[Precision editing of an aggression-encoding network relay suppresses violent action in mice](https://www.biorxiv.org/content/10.1101/2022.12.07.519272v4)**, by Yael S. Grossman<sup>*</sup>, Austin Talbot<sup>*</sup>, Neil M. Gallagher, Kathryn K. Walder-Christensen, Gwenaëlle E. Thomas, Alexandra Fink Skular, Scott J. Russo, David E. Carlson<sup>†</sup>, and Kafui Dzirasa<sup>†</sup>.

<sup>*</sup> Joint first authors; these authors contributed equally.

<sup>†</sup> Joint senior authors; these authors jointly supervised this work.

## Start here

| Analysis | Location |
| --- | --- |
| NMF implementations and historical training/projection analyses | [Network analysis](analysis/network) |
| Component selection and single-/two-region comparisons | [Regional analysis](analysis/regional) |
| Specific source locations and verification limits | [Analysis index](docs/analysis_index.md) |
| Closed-loop network mediation and regional theta controls, Figure 5e–g | [closed_loop_mediation.ipynb](analysis/mediation/closed_loop_mediation.ipynb) |
| ESR1/CNO mediation, Supplementary Figure S8 | [esr1_mediation.ipynb](analysis/mediation/esr1_mediation.ipynb) |
| Inputs, methods, and run instructions for mediation | [Mediation README](analysis/mediation/README.md) |

## Organization and naming

```text
analysis/
  network/
    models/
    experiments/
    thresholds/
    projections/
    joint_models/
    feature_experiments/
    archive_sources/
  regional/
    component_selection/
    single_region/
    two_region/
  mediation/
    closed_loop_mediation.ipynb
    esr1_mediation.ipynb
docs/
  analysis_index.md
  source_manifest.json
tools/
  check_release.py
  clear_notebook_outputs.py
```

## LFP preprocessing

The existing MATLAB preprocessing and feature-generation pipeline is maintained in [carlson-lab/lpne-data-analysis](https://github.com/carlson-lab/lpne-data-analysis). Its entry points include [runFeaturePipeline.m](https://github.com/carlson-lab/lpne-data-analysis/blob/d5b29a45278a5de567001ffe29af17e6a5189e1e/runFeaturePipeline.m), [preprocessData.m](https://github.com/carlson-lab/lpne-data-analysis/blob/d5b29a45278a5de567001ffe29af17e6a5189e1e/preprocessData.m), and [saveFeatures.m](https://github.com/carlson-lab/lpne-data-analysis/blob/d5b29a45278a5de567001ffe29af17e6a5189e1e/saveFeatures.m).

The learning framework is also available in [carlson-lab/encodedSupervision](https://github.com/carlson-lab/encodedSupervision).

## Running analyses

Start with the network and regional analyses listed in the [analysis index](docs/analysis_index.md), followed by the [mediation instructions](analysis/mediation/README.md). Inputs must be obtained separately. The manuscript directs data requests to Kafui Dzirasa (`kafui.dzirasa@duke.edu`) or David Carlson (`david.carlson@duke.edu`).

The network-modeling code retains its original environment and path assumptions. The manuscript describes Python 3.7 and TensorFlow 2.4 for model learning; this is historical context, not a tested modern installation recipe. Some scripts refer to local modules or fitted-model files that are not in this release. See the [analysis index](docs/analysis_index.md) before choosing an entry point.

Mediation p-values are simulation-based and can vary across random seeds. The manuscript reports values from the original analysis run.

## License

[GPL-3.0](LICENSE).
