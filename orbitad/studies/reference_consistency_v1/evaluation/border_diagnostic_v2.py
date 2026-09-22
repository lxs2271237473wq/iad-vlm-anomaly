"""Exact border strata v2, not directly comparable to the old touching rule."""
import sys
from pathlib import Path
from evaluation_common_v2 import run, aupro, edge_distance, stratum_for
ROOT = Path('/root/private_data/iad-vlm-anomaly')
DATA = ROOT / 'datasets/MVTec_AD_2'
META = ROOT / 'orbitad/results/a10_multilayer_v1/public_meta.csv'
OUT = ROOT / 'orbitad/results/tiny_ad2_study_v1/reassessment_v2/border'
A74 = ROOT / 'ad2_model_zoo/results/a74_ad2_matched_dual_resolution_v1'
CATS = ('can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts')
if __name__ == '__main__':
    run({'baseline': A74}, DATA, META, OUT, sys.argv[1:] or CATS, border=True)
