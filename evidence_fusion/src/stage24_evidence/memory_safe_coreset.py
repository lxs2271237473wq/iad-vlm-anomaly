"""CPU-backed embeddings, GPU projection and unchanged greedy selection rule.

Pass a preallocated CPU tensor (optionally backed by a memory-mapped file).
This avoids the original GPU embedding list plus vstack allocation.
"""
import torch
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy


@torch.inference_mode()
def select_cpu_backed(embedding, sampling_ratio=0.1, chunk_size=8192, device='cuda'):
    if embedding.device.type != 'cpu' or embedding.ndim != 2:
        raise ValueError('Expected a two-dimensional CPU embedding store')
    sampler = KCenterGreedy(embedding, sampling_ratio)
    sampler.model.fit(embedding)
    sampler.model.sparse_random_matrix = sampler.model.sparse_random_matrix.to(device)
    sampler.features = torch.empty((len(embedding), sampler.model.n_components), device=device)
    for start in range(0, len(embedding), chunk_size):
        end = min(start + chunk_size, len(embedding))
        sampler.features[start:end] = sampler.model.transform(embedding[start:end].to(device))
    sampler.reset_distances()
    idx = torch.randint(high=len(embedding), size=(1,), device=device).squeeze()
    selected = [int(idx.item())]
    for _ in range(sampler.coreset_size - 1):
        sampler.update_distances(idx)
        idx = sampler.get_new_idx()
        sampler.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
        selected.append(int(idx.item()))
    return selected


if __name__ == '__main__':
    import json
    from run_baselines import seed_all
    seed_all(123)
    x = torch.randn(2048, 1536)
    seed_all(42)
    expected = KCenterGreedy(x.cuda(), 0.1).select_coreset_idxs()
    seed_all(42)
    actual = select_cpu_backed(x, 0.1, chunk_size=256)
    assert expected == actual, 'Selected centers differ from original implementation'
    print(json.dumps({'status': 'passed', 'synthetic_shape': list(x.shape),
                      'selected_centers': len(actual), 'indices_identical': True,
                      'full_512_training_run': False}))
