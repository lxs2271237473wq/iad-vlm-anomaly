#!/usr/bin/env bash
set -euo pipefail
cd /root/private_data/iad-vlm-anomaly
/opt/conda/bin/python -u experiments/stage24_evidence/probe_mild_intervention.py
/opt/conda/bin/python -u experiments/stage24_evidence/summarize_mild_intervention.py
