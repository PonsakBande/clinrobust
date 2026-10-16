[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23399856.svg)](https://doi.org/10.5281/zenodo.23399856)

# clinrobust

## Reproducible robustness audits for clinical machine-learning models

`clinrobust` is a reproducible toolkit for evaluating clinical prediction
models under deployment-relevant input degradation:

1. **Modality ablation** — for example, complete absence of neuroimaging.
2. **Progressive missingness** — MCAR missingness with mean, median, and KNN imputation.
3. **Measurement noise** — feature-specific Gaussian-noise sensitivity analysis.
4. **Minimum viable input analysis** — backward elimination evaluated using repeated stratified cross-validation.

The toolkit was developed for the study:

> Protocol-dependent robustness of machine-learning Alzheimer's disease
> staging to missing neuroimaging and degraded clinical inputs: a simulation study.

## Key design guarantees

- **Config-driven analysis:** analyse a new dataset by creating a new YAML configuration file.
- **No test-data leakage:** imputers are always fitted on training data only.
- **Standardised KNN:** features are scaled before KNN distances are calculated.
- **Deterministic random seeds:** all simulations use explicit integer seeds.
- **Publication-ready outputs:** figures are automatically saved as 300-dpi PNG and vector PDF.
- **Stratified bootstrap:** preserves minority-class counts in bootstrap samples.

## Repository layout

```text
clinrobust/
├── clinrobust/
│   ├── config.py        # configuration handling
│   ├── data.py          # load, filter, encode, target generation, splitting
│   ├── models.py        # model pipeline, tuning, metrics, bootstrap CIs
│   ├── robustness.py    # MCAR, imputation, noise, feature elimination
│   └── reporting.py     # publication-quality figures and tables
├── config.yaml          # OASIS-2 reference configuration
├── run_audit.py         # end-to-end audit runner
├── requirements.txt
├── verification_report.csv
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/PonsakBande/clinrobust.git
cd clinrobust
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

1. Download the OASIS-2 longitudinal CSV from https://www.oasis-brains.org/.
2. Do not upload or redistribute OASIS data in this repository.
3. Edit `config.yaml` so `data.path` points to your local OASIS-2 CSV.
4. Run the full audit:

```bash
python run_audit.py
```

For a quick model-only smoke test:

```bash
python run_audit.py --skip-slow
```

## Analysing a new dataset

Copy `config.yaml`, for example:

```bash
cp config.yaml config_oasis3.yaml
python run_audit.py --config config_oasis3.yaml
```

Update the following fields in the new configuration:

- `data.path`
- `data.filters`
- `data.rename`
- `data.encode`
- `data.clinical_features`
- `data.imaging_features`
- `data.target.rules`

## External validation example: OASIS-3

Version 0.1.1 ships `config_oasis3.yaml`, the configuration used for the OASIS-3
external validation in the companion paper (n = 1,300; 89 AD cases). OASIS-3 must
first be harmonised to the OASIS-2 feature schema (one baseline row per subject with
Age, Gender, EDUC, MMSE, eTIV, nWBV, ASF and CDR); the procedure is described in the
paper's Methods. Edit `data.path` and `output.base_dir`, then:

```bash
python run_audit.py --config config_oasis3.yaml
```

### Changes in v0.1.1
- **Fix:** dataset-describing mappings (`filters`, `rename`, `encode`, `param_grid`)
  in a user YAML now *replace* the defaults instead of being merged. Previously
  `filters: {}` silently inherited the OASIS-2 `Visit: 1` filter and `encode`
  re-mapped already-numeric columns to NaN.
- Added `config_oasis3.yaml` and `CHANGELOG.md`.
- Re-verified: all 116 OASIS-2 checks pass under v0.1.1.

## Verification

The file `verification_report.csv` records 116 automated checks.
All checks passed when reproducing the OASIS-2 study results using:

- Python 3.13
- scikit-learn 1.6.1
- pandas 2.2.3
- NumPy 2.1.3

## Limitations

This toolkit demonstrates robustness auditing; it does not establish
clinical utility. The reference study used OASIS-2 (n=150; 13 AD cases),
simulated MCAR missingness, additive Gaussian noise, and one model family
(class-weighted Random Forest). Results must be externally validated before
clinical deployment.

## Data-use statement

No OASIS raw data are included in this repository. Users must obtain OASIS
datasets directly from the official data providers and comply with all data-use agreements.

## Citation

Software DOI (all versions): https://doi.org/10.5281/zenodo.23399856  
Code: https://github.com/PonsakBande/clinrobust

## License

MIT License. See `LICENSE`.
