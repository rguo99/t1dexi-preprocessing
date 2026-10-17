# T1DEXI Data Preprocessing and Instance Curation Pipeline

Data preprocessing for the paper:

> Guo R, Levine ME, Wu D, Zaharieva DP, Lal RA, Fox EB, Johari R.
> Clinical and Statistical Metrics Reveal Tradeoffs in Glucose Forecasting for
> Type 1 Diabetes Decision Support. 2026.

Code developed by Rachel Guo.

Turns the raw [T1DEXI](https://doi.org/10.25934/PR00008428.1) (Type 1 Diabetes
Exercise Initiative) dataset into curated 8-hour CGM / insulin / meal instances
with a per-user chronological train/test split.

It was written for this paper, whose experiment code is in
[glucose-forecasting-tradeoffs](https://github.com/rguo99/glucose-forecasting-tradeoffs)
and reads this pipeline's output. The output does not depend on that code, so
it can also be used to train and evaluate other glucose forecasting models on
T1DEXI.


## Data access

T1DEXI is available through [Vivli](https://vivli.org) under a data use
agreement. This repository contains code only. It contains no participant
data, and its outputs must not be redistributed outside the terms of your own
agreement. Download the raw SAS transport files (`*.xpt`) into one folder; the
pipeline reads `FACM`, `LB`, `ML`, `FAMLPM`, `DX`, `PR` and `DM`.

## Requirements

- Python 3.11 (tested with 3.11.14) on Linux.
- CPU only; no GPU is needed.
- Packages pinned in `requirements.txt`:

  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Resources for the full cohort (all 502 participants): 16 CPU cores and 64 GB
  of memory; about 20 minutes end to end. The seven raw domains take ~5 GB and
  the outputs ~2 GB.

## Quick start

```bash
export T1DEXI_RAW_DIR=/path/to/T1DEXI        # folder with the .xpt files
export T1DEXI_OUTPUT_DIR=/path/to/output     # optional; default ./output
export T1DEXI_NUM_CPUS=16                    # optional; default = available CPUs, max 16
bash run_pipeline.sh
```

On a Slurm cluster, edit the partition, paths and environment activation in
`slurm/run_pipeline.sbatch`, then run `sbatch slurm/run_pipeline.sbatch` from the
repository root. Steps whose outputs already exist are skipped, so an
interrupted run can be resumed; delete `$T1DEXI_OUTPUT_DIR` to start from scratch.
`python config.py` prints the resolved paths and which inputs exist.

## Pipeline steps

| # | Script | What it does | Main outputs (under `$T1DEXI_OUTPUT_DIR`) |
|---|---|---|---|
| 1 | `convert_xpt_to_parquet.py` | One-time `.xpt` to parquet conversion (written next to the `.xpt` files) | `$T1DEXI_RAW_DIR/*.parquet` |
| 2 | `get_netiob_cleaned_data.py` | Cleans the glucose, meal, insulin, exercise and device domains (netIOB utilities); keeps only consumed meal carbohydrate | `netiob_cleaned_data/*.parquet`, `datasets/exercise_data/exercise.csv` |
| 3 | `process_insulin.py` | Removes known insulin-record anomalies and converts basal delivery to a 5-minute grid (U per 5-min slot) for every device type | `cleaned_data/basal.parquet` |
| 4 | `build_carb_lookup.py` | Consumed carbohydrate per meal event from FAMLPM | `cleaned_data/carb_consumed_lookup.parquet` |
| 5 | `split_parquets_by_user.py` | Splits CGM, meal, bolus and basal tables by participant | `by_user/<feature>/<user>.parquet` |
| 6 | `instance_feature_extraction.py` | 8-hour non-exercise windows (5-minute stride) and the records inside each window | `processed_data/non_exercise_8h/non_exercise.csv`, `.../by_user/` |
| 7 | `combine_by_user.py` | Combined per-feature tables | `processed_data/non_exercise_8h/{cgm,meals,bolus,basal}.parquet`, `meals.csv`, `bolus.csv` |
| 8 | `curate_data.py` | Quality filter, chronological 70/30 split with 8-hour washout, balanced selection of up to 150 train / 150 test instances per user | `data_filtering/non_exercise_8h/` |

The other files: `config.py` (paths and settings, read from the environment
variables above), `ultra_fast_features_per_user.py` (a helper called by step 6),
`netiob/` (the adapted netIOB utilities used by steps 2, 3 and 6),
`slurm/run_pipeline.sbatch` and `figures/` (see Figure below).

### Processing details

- **Insulin (step 3).** Extends the netIOB automated-insulin-delivery chunking
  to every device type (AID pumps, other pumps, MDI). Basal is in units per
  5-minute slot; a slot is 0 during a pump suspension and NaN when no record
  covers it. Six participants and specific dates for 14 others are removed
  following the netIOB data-quality review (listed in `process_insulin.py`).
- **Meals (steps 2 and 4).** FAMLPM records each meal as TAKEN, RETURNED and
  CONSUMED; only CONSUMED amounts are used.
- **Windows (step 6).** Windows overlapping an exercise session or the 2 hours
  after one are excluded. Instance ids (`ex`) are deterministic across reruns.
- **Curation (step 8).** Instances need at least 96 CGM readings and are
  labelled meal or non-meal around the forecast origin (the 48th reading). Each
  user's instances are split 70/30 in time with an 8-hour washout and balanced
  50/50 meal / non-meal, up to 150 per split. Thresholds are in `curate_data.py`.

## Outputs used downstream

```
$T1DEXI_OUTPUT_DIR/
  netiob_cleaned_data/user_device_data.parquet  # USUBJID, DXTRT, DEVICE, device_type
  processed_data/non_exercise_8h/
    non_exercise.csv                            # ex, id, time_series_start, time_series_end
    by_user/{cgm,meals,bolus,basal}/<user>.parquet
    {cgm,meals,bolus,basal}.parquet, meals.csv, bolus.csv
  data_filtering/non_exercise_8h/
    train_instances.pkl, test_instances.pkl     # flat lists of instance ids
    filtered_user_instances.pkl                 # {user_id: train_ids + test_ids}
    per_user/user_<id>.pkl                      # {user_id, train_ids, test_ids}
```

To rebuild one instance, take its `ex` id from the split lists, load the
user's `by_user/<feature>/<user>.parquet` files and select rows with that `ex`.

### Cohort size

On the full T1DEXI release the pipeline keeps 487 of 502 participants
(144,972 instances: 72,823 train / 72,149 test). It does not exclude anyone by
treatment; 88 of those participants use multiple daily injections (MDI). Use
`user_device_data.parquet` (`DEVICE == "non-AID (MDI)"`) to drop them if your
model assumes pump insulin.

The paper's 395-user cohort is selected downstream, by
`eval_config.load_eligible_users` in glucose-forecasting-tradeoffs: pump users
(the 88 MDI users excluded) with at least 150 train and 10 test instances and
no duplicate CGM timestamps (2 users excluded).

## Figure

`figures/instance_gap_histogram.py` plots the gaps between chronologically
adjacent curated instances (the instance-gap figure in the paper's Methods). Pass
`--metrics-df` (or `--user-ids`) to restrict it to the evaluated users.

## License and attribution

MIT (see `LICENSE`). The code in `netiob/` is adapted from the netIOB T1DEXI
processing utilities by CeADAR (copied from the netIOB GitLab repository, not
the `netiob` PyPI package) and remains under the Apache License 2.0
(`netiob/LICENSE`, `netiob/NOTICE`). If you
use this pipeline, please cite the paper above and the netIOB paper:

> Lewis DM, et al. Data Processing and Best Practices in Large-Scale Diabetes
> Datasets. Preprints 2026. https://doi.org/10.20944/preprints202607.0522.v1

The netIOB code was obtained from its original GitLab repository,
<https://gitlab.com/CeADARIreland_Public/netiob>, which is no longer reachable. It was
copied from that source, not installed from PyPI. PyPI release
[`netiob` 0.1.1](https://pypi.org/project/netiob/0.1.1/) lists the same
maintainer, contributors, homepage (the same GitLab repository) and license, so
it is likely the same project, though the exact commit used here was never
confirmed byte-identical to that release.
