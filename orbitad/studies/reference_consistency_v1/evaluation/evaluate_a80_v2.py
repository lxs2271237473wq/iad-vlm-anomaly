"""A80 reassessment v2; CPU only, complete requested variants only."""
import sys
from pathlib import Path
from evaluation_common_v2 import run, aupro
ROOT = Path('/root/private_data/iad-vlm-anomaly')
DATA = ROOT / 'datasets/MVTec_AD_2'
META = ROOT / 'orbitad/results/a10_multilayer_v1/public_meta.csv'
OUT = ROOT / 'orbitad/results/tiny_ad2_study_v1/reassessment_v2/a80'
ROOTS = {
    'baseline': ROOT / 'ad2_model_zoo/results/a74_ad2_matched_dual_resolution_v1',
    'dense1792': ROOT / 'ad2_model_zoo/results/a80_dense_query_v1/long1792',
    'dense2240': ROOT / 'ad2_model_zoo/results/a80_dense_query_v1/long2240',
}
SRC = {'fabric': (2048,2448), 'sheet_metal': (1056,4224)}
def native_px_per_token(category, variant):
    h,w = SRC[category]
    return 14 * min(h,w)/672 if variant == 'baseline' else 14 * max(h,w)/int(variant.replace('dense',''))
if __name__ == '__main__':
    run(ROOTS, DATA, META, OUT, sys.argv[1:] or ['fabric', 'sheet_metal'])
