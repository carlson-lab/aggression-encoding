# Mediation analyses

These two notebooks contain the mediation analyses associated with Figure 5e–g and Supplementary Figure S8. They are distributed without data or saved outputs. Calculations have been retained from the original working notebooks; descriptive names, headings, and corrected comments make their scope explicit.

## Files and inputs

| Notebook | Original working name | Required input files |
| --- | --- | --- |
| [closed_loop_mediation.ipynb](closed_loop_mediation.ipynb) | `Aggression_Causal_Mediation_closed_alllaser_social_power.ipynb` | `ClosedRandomLoop_behavior&scores.mat`; `Closed_loop_singleregionMediation_fixed.mat` |
| [esr1_mediation.ipynb](esr1_mediation.ipynb) | `Aggression_Causal_Mediation_ESR1_just_female.ipynb` | `ESR1_labelsScoresPower.mat` |

Input filenames are unchanged because they are referenced directly by the notebooks. Place authorized local copies beside the notebooks; `.gitignore` excludes them. No input data, model weights, or manuscript attachments are included here.

## Environment and execution

Use a separate environment for mediation; TensorFlow is not required for these two notebooks. `requirements.txt` lists a starting environment, not a reconstruction of the original environment lockfile. NumPy is bounded below version 2 because the retained ESR1 source uses `np.NaN`.

From the repository root:

```bash
python3 -m venv .venv-mediation
source .venv-mediation/bin/activate
python -m pip install -r analysis/mediation/requirements.txt
cd analysis/mediation
jupyter notebook
```

Open the desired notebook and restart the kernel before running all cells. Dataset access is required; the source-only release has not been executed end-to-end. Before committing any rerun, clear outputs with `python ../../tools/clear_notebook_outputs.py` and run `python ../../tools/check_release.py`.

## Closed-loop analysis

The notebook selects blue- and yellow-stimulated windows in the intact-male condition. Aggressive behavior is the positive outcome; non-aggressive interaction and non-interaction form the comparison outcome.

- **Figure 5e:** compare logistic regressions with stimulation alone versus stimulation plus network score; then fit network mediation using a probit-link binomial outcome model and an OLS mediator model.
- **Figure 5f:** substitute each region's theta power as the mediator.
- **Figure 5g:** retain the network mediator while including each region's theta power as a covariate in both mediator and outcome models.
- **Additional exploratory model:** the final cell includes all eleven regional covariates simultaneously. It is not the per-region analysis in Figure 5g.

Mediation uses `statsmodels.stats.Mediation(...).fit()`. The likelihood-ratio comparison uses the upper tail of a chi-square distribution; it should not be described as a two-tailed effect test. The closed-loop notebook does not set an explicit random seed. The author's manuscript results are from the original run; the later saved rerun is not an exact-results reference for this release.

## ESR1 analysis

The active condition filter includes both intact-male and female encounters, despite the original `just_female` filename. Treatment is CNO versus saline, with network activity as mediator and aggressive versus non-aggressive/non-interaction behavior as outcome.

The notebook includes an exploratory mixed-effects mediator fit and an unadjusted mediation fit. The subsequent mouse-adjusted mediation actually uses **mouse-indicator covariates** in OLS/probit GLM models; the separate mixed-effects fit is not passed to `Mediation`.

The last analysis retains the original `np.convolve(..., np.ones(60), 'same')` operation. This computes a **centered 60-row sum on the concatenated, filtered data**, without mouse/session grouping. It is not a trailing, per-session average. The README describes the code as implemented rather than repeating the supplement's different wording; no smoothing calculation was silently changed during packaging. The original seed of 42 is preserved.

## Verification boundary

Notebook schemas, output removal, syntax, and preservation of executable logic were checked. These checks do not establish agreement with every numerical manuscript result or validate the scientific assumptions of the mediation models. Any future change to treatment coding, covariates, smoothing, or sampling should be reviewed as an analysis change rather than a naming cleanup.
