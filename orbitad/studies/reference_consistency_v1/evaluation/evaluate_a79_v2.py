"""A79 reassessment v2; CPU only, complete requested variants only."""
import sys
from pathlib import Path
from evaluation_common_v2 import run, aupro
ROOT = Path('/root/private_data/iad-vlm-anomaly')
DATA = ROOT / 'datasets/MVTec_AD_2'
META = ROOT / 'orbitad/results/a10_multilayer_v1/public_meta.csv'
OUT = ROOT / 'orbitad/results/tiny_ad2_study_v1/reassessment_v2/a79'
ROOTS = {
    'baseline': ROOT / 'ad2_model_zoo/results/a74_ad2_matched_dual_resolution_v1',
    'brightness': ROOT / 'ad2_model_zoo/results/a79_ref_illumination_v1/brightness',
    'directional': ROOT / 'ad2_model_zoo/results/a79_ref_illumination_v1/directional',
}
if __name__ == '__main__':
    run(ROOTS, DATA, META, OUT, sys.argv[1:] or ['sheet_metal', 'can'])
