#!/usr/bin/env bash
# Run the full T1DEXI preprocessing pipeline, in order.
#
#   export T1DEXI_RAW_DIR=/path/to/T1DEXI          # folder with the raw .xpt files
#   export T1DEXI_OUTPUT_DIR=/path/to/output        # optional, default ./output
#   export T1DEXI_NUM_CPUS=16                       # optional, default: available CPUs (max 16)
#   bash run_pipeline.sh
#
# Steps that find their outputs already present are skipped, so the script can
# be rerun after an interruption. Delete $T1DEXI_OUTPUT_DIR to start over.
set -euo pipefail
cd "$(dirname "$0")"

: "${T1DEXI_RAW_DIR:?Set T1DEXI_RAW_DIR to the folder containing the raw T1DEXI .xpt files}"

# One thread per worker process; parallelism comes from multiprocessing / Ray.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

step() { echo; echo "### [$(date '+%F %T')] $*"; }

step "1/8 convert .xpt to parquet";        python convert_xpt_to_parquet.py
step "2/8 clean raw domains (netIOB)";     python get_netiob_cleaned_data.py
step "3/8 5-minute basal insulin";         python process_insulin.py
step "4/8 consumed-carbohydrate lookup";   python build_carb_lookup.py
step "5/8 split by user";                  python split_parquets_by_user.py
step "6/8 instance windows + features";    python instance_feature_extraction.py
step "7/8 combine per-user features";      python combine_by_user.py
step "8/8 curate train/test instances";    python curate_data.py

echo
echo "Done. Outputs in ${T1DEXI_OUTPUT_DIR:-$(pwd)/output}"
