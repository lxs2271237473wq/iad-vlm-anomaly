import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch
import json
from sklearn.cluster import KMeans
from src.utils import augment_image, dists2map, min_max_norm, cvt2heatmap, heatmap_on_image
from src.post_eval import mean_top1p
from src.sampler import GreedyCoresetSampler

def fill_closed_regions(image):
    if image is None:
        print("Invalid input image")
        return None

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(image)
    cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    return filled

def _directional_gain(shape, rng, strength=0.35):
    """Smooth linear illumination ramp across a random direction.

    Models a change in illumination *direction* (the AD2 shift_* axis), not a
    change in overall brightness.
    """
    height, width = shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = xx / max(width - 1, 1) - 0.5
    yy = yy / max(height - 1, 1) - 0.5
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    ramp = np.cos(theta) * xx + np.sin(theta) * yy
    ramp = ramp / max(float(np.abs(ramp).max()), 1e-6)
    return (1.0 + strength * ramp)[..., None]


def augment_reference_illumination(img_rgb, mode, seed):
    """Extra reference views along a chosen illumination axis.

    none        -> no extra views
    brightness  -> two multiplicative brightness views (the SuperADD axis)
    directional -> two directional illumination views (the measured AD2 axis)
    """
    if mode in ("", "none"):
        return []
    rng = np.random.default_rng(seed)
    out = []
    if mode == "brightness":
        for factor in (0.8, 1.2):
            out.append(np.clip(img_rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8))
    elif mode == "directional":
        for strength in (0.25, 0.40):
            gain = _directional_gain(img_rgb.shape, rng, strength)
            out.append(np.clip(img_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    else:
        raise ValueError(f"unknown SUPERAD_REF_AUG mode: {mode}")
    return out


def run_anomaly_detection_multilayer(
        model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        device,
        save_examples=False,
        masking=None,
        mask_ref_images=False,
        rotation=False,
        knn_metric='L2_normalized',
        knn_neighbors=1,
        faiss_on_cpu=False,
        seed=0,
        save_patch_dists=True,
        save_tiffs=False):
    """
    Updated to support multi-layer feature extraction and layer-wise knn matching.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    type_anomalies = list(set(object_anomalies[object_name] + ['good']))

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    img_ref_samples = sorted(os.listdir(img_ref_folder))

    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
        n_ref_samples = len(img_ref_samples)
        
    ######################################## Random selection ########################################    
    # if n_ref_samples != -1:
    #     img_ref_samples = img_ref_samples[seed * n_ref_samples:(seed + 1) * n_ref_samples]

    ####################################### Coreset selection ########################################   
    # Extract CLS features for all reference images
    cls_features = []
    valid_img_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img_tensor, _ = model.prepare_image(img_rgb)
            cls_feats = model.extract_cls_features(img_tensor)
            cls_features.append(cls_feats.squeeze().cpu())
            valid_img_names.append(img_name)

    cls_features = torch.stack(cls_features).to(device)  # Shape: (n_samples, 1024)
    sampler = GreedyCoresetSampler(percentage=0.1, device=device, dimension_to_project_features_to=1024)
    selected_indices = sampler.run(cls_features)
    
    # Select the corresponding image names
    img_ref_samples = [valid_img_names[idx] for idx in selected_indices]

    ######################################## K-Means selection ########################################
    # # Extract CLS features for all reference images
    # cls_features = []
    # valid_img_names = []
    # with torch.inference_mode():
    #     for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
    #         image_path = os.path.join(img_ref_folder, img_name)
    #         img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #         img_tensor, _ = model.prepare_image(img_rgb)
    #         cls_feats = model.extract_cls_features(img_tensor)
    #         cls_features.append(cls_feats.cpu().numpy().squeeze())
    #         valid_img_names.append(img_name)

    # cls_features = np.array(cls_features)  # Shape: (n_samples, 1024)

    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=n_ref_samples, random_state=seed, n_init=10)
    # kmeans.fit(cls_features)

    # # Find the image closest to each cluster center
    # cluster_centers = kmeans.cluster_centers_
    # selected_indices = []
    # for center in cluster_centers:
    #     distances = np.linalg.norm(cls_features - center, axis=1)
    #     closest_idx = np.argmin(distances)
    #     selected_indices.append(closest_idx)

    # # Select the corresponding image names
    # img_ref_samples = [valid_img_names[idx] for idx in selected_indices]
    ####################################################################################################
    
    
    feature_refs = {}  # {layer_name: [features]}
    knn_indices = {}   # {layer_name: faiss_index}
    grid_size = None

    with torch.inference_mode():
        start_time = time.time()

        for img_name in tqdm(img_ref_samples, desc="Extracting reference features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            aug_images = list(augment_image(img_rgb) if rotation else [img_rgb])
            ref_aug_mode = os.environ.get("SUPERAD_REF_AUG", "none").strip()
            if ref_aug_mode and ref_aug_mode != "none":
                aug_images.extend(augment_reference_illumination(
                    img_rgb, ref_aug_mode,
                    seed=abs(hash((img_name, ref_aug_mode, seed))) % (2**32)))
            for aug in aug_images:
                img_tensor, grid_size = model.prepare_image(aug)
                feats_dict = model.extract_features(img_tensor)
                idx=0
                mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1,
                                                         masking_type=(mask_ref_images and masking))
                for feats in feats_dict:
                    selected_feats = feats[mask]
                    if f'layer{idx}' not in feature_refs:
                        feature_refs[f'layer{idx}'] = []
                    feature_refs[f'layer{idx}'].append(selected_feats)
                    idx+=1

        # Concatenate and build FAISS index for each layer
        for layer_name, feats_list in feature_refs.items():
            layer_feats = np.concatenate(feats_list, axis=0).astype('float32')
            if knn_metric == 'L2_normalized':
                faiss.normalize_L2(layer_feats)

            if faiss_on_cpu:
                index = faiss.IndexFlatL2(layer_feats.shape[1])
            else:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, layer_feats.shape[1])

            index.add(layer_feats)
            knn_indices[layer_name] = index

        time_memorybank = time.time() - start_time

        inference_times = {}
        anomaly_scores = {}

        tiled_query = os.environ.get("SUPERAD_TILED_QUERY", "0") == "1"
        tiled_only = os.environ.get("SUPERAD_TILED_ONLY", "0") == "1"
        tiled_resolution = int(os.environ.get("SUPERAD_TILED_RESOLUTION", "672"))
        residual_weight = float(os.environ.get("SUPERAD_TILED_RESIDUAL_WEIGHT", "0.10"))

        def infer_map(img_rgb, resolution=None):
            """Run the shared memory bank on one image/crop at a chosen shorter edge."""
            previous_resolution = model.smaller_edge_size
            if resolution is not None:
                if hasattr(model, "set_smaller_edge_size"):
                    model.set_smaller_edge_size(resolution)
                else:
                    model.smaller_edge_size = resolution
            try:
                img_tensor, query_grid_size = model.prepare_image(img_rgb)
                feats_dict = model.extract_features(img_tensor)
                query_mask = model.compute_background_mask(
                    feats_dict[0], query_grid_size, threshold=1, masking_type=masking)
                maps = []
                for num, feats in enumerate(feats_dict):
                    masked_feats = feats[query_mask]
                    if knn_metric == "L2_normalized":
                        faiss.normalize_L2(masked_feats)
                    dists, _ = knn_indices[f'layer{num}'].search(masked_feats, k=knn_neighbors)
                    if knn_neighbors > 1:
                        dists = dists.mean(axis=1)
                    dists = 1 - dists
                    dmap = np.zeros_like(query_mask, dtype=float)
                    dmap[query_mask] = dists.squeeze()
                    dmap = dmap.reshape(query_grid_size)
                    maps.append(cv2.resize(dmap, (img_rgb.shape[1], img_rgb.shape[0])))
                stats_path = os.environ.get("SUPERAD_LAYER_STATS", "").strip()
                if stats_path:
                    import json as _json
                    with open(stats_path, "a") as _handle:
                        _handle.write(_json.dumps({
                            "resolution": int(model.smaller_edge_size),
                            "layers": [
                                {"q95": float(np.quantile(_m, 0.95)),
                                 "q99": float(np.quantile(_m, 0.99)),
                                 "max": float(_m.max()),
                                 "mean": float(_m.mean())}
                                for _m in maps
                            ],
                            "averaged_q99": float(np.quantile(np.mean(maps, axis=0), 0.99)),
                        }) + "\n")
                return np.mean(maps, axis=0)
            finally:
                if hasattr(model, "set_smaller_edge_size"):
                    model.set_smaller_edge_size(previous_resolution)
                else:
                    model.smaller_edge_size = previous_resolution

        def infer_tiled_map(img_rgb):
            """Split along the long axis into non-overlapping square crops and stitch."""
            height, width = img_rgb.shape[:2]
            tile_size = min(height, width)
            long_size = max(height, width)
            starts = list(range(0, long_size, tile_size))
            if starts and starts[-1] + tile_size > long_size:
                starts[-1] = max(0, long_size - tile_size)
            starts = sorted(set(starts))
            accum = np.zeros((height, width), np.float32)
            counts = np.zeros((height, width), np.float32)
            for start in starts:
                if width >= height:
                    crop = img_rgb[:, start:min(start + tile_size, width)]
                    tile_map = infer_map(crop, tiled_resolution)
                    accum[:, start:start + crop.shape[1]] += tile_map
                    counts[:, start:start + crop.shape[1]] += 1
                else:
                    crop = img_rgb[start:min(start + tile_size, height), :]
                    tile_map = infer_map(crop, tiled_resolution)
                    accum[start:start + crop.shape[0], :] += tile_map
                    counts[start:start + crop.shape[0], :] += 1
            return accum / np.maximum(counts, 1)

        # --- A77 optional multi-component export (all flags default to off) ---
        # extra_global_resolutions: whole-image queries at additional resolutions.
        # local_query: 2D sliding-window local views, each window inferred at
        # tiled_resolution, averaged back over overlapping windows.
        extra_global_resolutions = [
            int(value) for value in
            os.environ.get("SUPERAD_EXTRA_GLOBAL_RESOLUTIONS", "").split(",")
            if value.strip()
        ]
        local_query = os.environ.get("SUPERAD_LOCAL_QUERY", "0") == "1"
        local_window_fraction = float(os.environ.get("SUPERAD_LOCAL_WINDOW_FRACTION", "0.5"))
        local_stride_fraction = float(os.environ.get("SUPERAD_LOCAL_STRIDE_FRACTION", "0.5"))
        max_test_images_env = os.environ.get("SUPERAD_MAX_TEST_IMAGES", "").strip()
        max_test_images = int(max_test_images_env) if max_test_images_env else None

        def window_starts(length, window, stride):
            """Sliding-window offsets; the last window is snapped to the far edge."""
            if window >= length:
                return [0]
            offsets = list(range(0, length - window + 1, stride))
            if offsets[-1] != length - window:
                offsets.append(length - window)
            return offsets

        def infer_local_maps(img_rgb):
            """2D sliding-window local views plus cross-window agreement statistics.

            Every pixel is covered by several overlapping windows. Alongside the
            stitched mean map this returns (a) the per-pixel standard deviation
            across the covering windows, which measures how much the windows
            disagree, (b) the mean of per-window q99-normalised maps, which removes
            window-to-window score-scale drift, and (c) the coverage count. All of
            it is a by-product of the inference that already happens, so the extra
            statistics cost no additional forward passes and let several fusion
            rules be compared offline.
            """
            height, width = img_rgb.shape[:2]
            window = max(1, int(round(min(height, width) * local_window_fraction)))
            stride = max(1, int(round(window * local_stride_fraction)))
            tops = window_starts(height, window, stride)
            lefts = window_starts(width, window, stride)
            accum = np.zeros((height, width), np.float32)
            accum_sq = np.zeros((height, width), np.float32)
            accum_norm = np.zeros((height, width), np.float32)
            counts = np.zeros((height, width), np.float32)
            center = np.zeros((height, width), np.float32)

            def ownership(length, offsets):
                """Split every pairwise overlap in half: each pixel is owned by the
                window whose centre is closest, so nothing is averaged. This is the
                assignment used by the SuperADD patched execution."""
                bounds = [0]
                for i in range(1, len(offsets)):
                    overlap_mid = (offsets[i] + offsets[i - 1] + window) / 2.0
                    bounds.append(int(np.ceil(overlap_mid)))
                bounds.append(length)
                return [(bounds[i], bounds[i + 1]) for i in range(len(offsets))]

            own_rows = ownership(height, tops)
            own_cols = ownership(width, lefts)

            for i, top in enumerate(tops):
                for j, left in enumerate(lefts):
                    crop = img_rgb[top:top + window, left:left + window]
                    tile_map = infer_map(crop, tiled_resolution)
                    rows = slice(top, top + window)
                    cols = slice(left, left + window)
                    accum[rows, cols] += tile_map
                    accum_sq[rows, cols] += tile_map * tile_map
                    window_q99 = float(np.quantile(tile_map, 0.99))
                    accum_norm[rows, cols] += tile_map / max(window_q99, 1e-6)
                    counts[rows, cols] += 1
                    # centre-owned copy of this window, with no blending
                    r0, r1 = own_rows[i]
                    c0, c1 = own_cols[j]
                    center[r0:r1, c0:c1] = tile_map[r0 - top:r1 - top, c0 - left:c1 - left]

            safe = np.maximum(counts, 1)
            mean = accum / safe
            variance = np.maximum(accum_sq / safe - mean * mean, 0.0)
            return {
                "mean": mean,
                "std": np.sqrt(variance),
                "norm": accum_norm / safe,
                "count": counts,
                "center": center,
            }

        # A80: dense single-pass query. Instead of sweeping square windows, run ONE
        # forward pass over the whole image sized so the long side is
        # SUPERAD_DENSE_LONG. infer_map already resizes the short side, so the
        # equivalent short-side target is long_side / aspect.
        dense_long = int(os.environ.get("SUPERAD_DENSE_LONG", "0") or 0)

        def infer_dense_map(img_rgb):
            height_d, width_d = img_rgb.shape[:2]
            aspect = max(height_d, width_d) / max(min(height_d, width_d), 1)
            short_target = max(14, int(round(dense_long / aspect)))
            return infer_map(img_rgb, short_target)

        requested_split = os.environ.get("SUPERAD_TEST_SPLIT", "").strip()
        test_split = requested_split or ("test_public" if os.path.isdir(f"{data_root}/{object_name}/test_public") else "test")
        available_types = [name for name in type_anomalies if os.path.isdir(f"{data_root}/{object_name}/{test_split}/{name}")]
        if not available_types:
            raise FileNotFoundError(f"No supported image folders in {data_root}/{object_name}/{test_split}")
        for anomaly_type in tqdm(available_types, desc=f"Processing {object_name}"):
            test_dir = f"{data_root}/{object_name}/{test_split}/{anomaly_type}"
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}", exist_ok=True)
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test_hm_on_img/{anomaly_type}", exist_ok=True)

            test_images = sorted(os.listdir(test_dir))
            if max_test_images is not None:
                test_images = test_images[:max_test_images]
            for idx, test_img_name in enumerate(test_images):
                start_time = time.time()
                test_path = os.path.join(test_dir, test_img_name)
                
                img_rgb = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
                global_map = None if (tiled_query and tiled_only) else infer_map(img_rgb)
                anomaly_map = global_map
                tiled_map = None
                if tiled_query:
                    tiled_map = infer_dense_map(img_rgb) if dense_long else infer_tiled_map(img_rgb)
                    if tiled_only:
                        anomaly_map = tiled_map
                    else:
                        global_q = float(np.quantile(global_map, .995))
                        tiled_q = float(np.quantile(tiled_map, .995))
                        scale = global_q / max(tiled_q, 1e-12)
                        anomaly_map = (1.0 - residual_weight) * global_map + residual_weight * tiled_map * scale
                anomaly_map_norm = min_max_norm(anomaly_map)
                score = mean_top1p(anomaly_map.flatten())

                inference_times[f"{anomaly_type}/{test_img_name}"] = time.time() - start_time
                anomaly_scores[f"{anomaly_type}/{test_img_name}"] = score

                if save_tiffs:
                    fname = os.path.splitext(test_img_name)[0]
                    if os.environ.get("SUPERAD_SAVE_OVERLAYS", "0") == "1":
                        heatmap = cvt2heatmap(anomaly_map_norm * 255)
                        hm_on_img = heatmap_on_image(heatmap, img_rgb)
                        cv2.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test_hm_on_img/{anomaly_type}/{fname}.jpg", hm_on_img)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}/{fname}.tiff", anomaly_map)
                    if tiled_query:
                        component_dir = f"{plots_dir}/component_maps/seed={seed}/{object_name}/{anomaly_type}"
                        os.makedirs(component_dir, exist_ok=True)
                        if global_map is not None:
                            tiff.imwrite(f"{component_dir}/{fname}_global.tiff", global_map)
                        tiff.imwrite(f"{component_dir}/{fname}_tiled.tiff", tiled_map)
                    if extra_global_resolutions or local_query:
                        component_dir = f"{plots_dir}/component_maps/seed={seed}/{object_name}/{anomaly_type}"
                        os.makedirs(component_dir, exist_ok=True)
                        # export-only: does not change anomaly_map or the reported score
                        for extra_resolution in extra_global_resolutions:
                            tiff.imwrite(
                                f"{component_dir}/{fname}_global{extra_resolution}.tiff",
                                infer_map(img_rgb, extra_resolution),
                            )
                        if local_query:
                            local_maps = infer_local_maps(img_rgb)
                            tiff.imwrite(
                                f"{component_dir}/{fname}_local{tiled_resolution}.tiff",
                                local_maps["mean"],
                            )
                            tiff.imwrite(
                                f"{component_dir}/{fname}_localstd{tiled_resolution}.tiff",
                                local_maps["std"],
                            )
                            tiff.imwrite(
                                f"{component_dir}/{fname}_localnorm{tiled_resolution}.tiff",
                                local_maps["norm"],
                            )
                            tiff.imwrite(
                                f"{component_dir}/{fname}_localcenter{tiled_resolution}.tiff",
                                local_maps["center"],
                            )
                if save_patch_dists:
                    np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}/{test_img_name.split('.')[0]}.npy", anomaly_map)

                if save_examples and idx < 3:
                    num_layers = 0
                    cols = 3
                    rows = int(np.ceil((3 + num_layers) / cols))

                    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
                    axes = axes.flatten()

                    # 原始图像
                    axes[0].imshow(img_rgb)
                    axes[0].set_title("Test Image")
                    axes[0].axis("off")

                    # 平均 anomaly map
                    im = axes[1].imshow(anomaly_map_norm, cmap='jet')
                    axes[1].set_title("Avg Anomaly Map")
                    axes[1].axis("off")
                    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, orientation="horizontal")

                    # Histogram
                    axes[2].hist(anomaly_map.flatten(), bins=50)
                    axes[2].axvline(score, color='red', linestyle='dashed')
                    axes[2].set_title("Score Histogram")

                    # heatmaps by layers
                    for i, d_masked in enumerate([]):
                        ax = axes[3 + i]
                        im = ax.imshow(d_masked, cmap='jet')
                        ax.set_title(f"Layer {i} Map")
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")

                    for j in range(3 + num_layers, len(axes)):
                        axes[j].axis("off")

                    plt.tight_layout()
                    example_dir = f"{plots_dir}/{object_name}/examples"
                    os.makedirs(example_dir, exist_ok=True)
                    plt.savefig(f"{example_dir}/example_{anomaly_type}_{idx}.png")
                    plt.close()


    return anomaly_scores, time_memorybank, inference_times

def run_anomaly_detection_multilayer_private(
        model,
        object_name,
        data_root,
        n_ref_samples,
        plots_dir,
        save_examples=False,
        masking=None,
        mask_ref_images=False,
        rotation=False,
        knn_metric='L2_normalized',
        knn_neighbors=1,
        faiss_on_cpu=False,
        seed=0,
        save_tiffs=False):
    """
    Updated to support multi-layer feature extraction and layer-wise knn matching.
    """

    assert knn_metric in ["L2", "L2_normalized"]

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    img_ref_samples = sorted(os.listdir(img_ref_folder))
    # if n_ref_samples != -1:
    #     img_ref_samples = img_ref_samples[seed * n_ref_samples:(seed + 1) * n_ref_samples]

    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
        n_ref_samples = len(img_ref_samples)
        
    ######################################## Coreset selection ########################################   
    # Extract CLS features for all reference images
    cls_features = []
    valid_img_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img_tensor, _ = model.prepare_image(img_rgb)
            cls_feats = model.extract_cls_features(img_tensor)
            cls_features.append(cls_feats.squeeze().cpu())
            valid_img_names.append(img_name)

    cls_features = torch.stack(cls_features).to(device)  # Shape: (n_samples, 1024)
    sampler = GreedyCoresetSampler(percentage=0.1, device=device, dimension_to_project_features_to=1024)
    selected_indices = sampler.run(cls_features)
    
    # Select the corresponding image names
    img_ref_samples = [valid_img_names[idx] for idx in selected_indices]
    
    ######################################## K-Means selection ########################################
    # # Extract CLS features for all reference images
    # cls_features = []
    # valid_img_names = []
    # with torch.inference_mode():
    #     for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
    #         image_path = os.path.join(img_ref_folder, img_name)
    #         img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #         img_tensor, _ = model.prepare_image(img_rgb)
    #         cls_feats = model.extract_cls_features(img_tensor)
    #         cls_features.append(cls_feats.cpu().numpy().squeeze())
    #         valid_img_names.append(img_name)

    # cls_features = np.array(cls_features)  # Shape: (n_samples, 1024)

    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=n_ref_samples, random_state=seed, n_init=10)
    # kmeans.fit(cls_features)

    # # Find the image closest to each cluster center
    # cluster_centers = kmeans.cluster_centers_
    # selected_indices = []
    # for center in cluster_centers:
    #     distances = np.linalg.norm(cls_features - center, axis=1)
    #     closest_idx = np.argmin(distances)
    #     selected_indices.append(closest_idx)

    # # Select the corresponding image names
    # img_ref_samples = [valid_img_names[idx] for idx in selected_indices]
    ####################################################################################################    
    

    feature_refs = {}  # {layer_name: [features]}
    knn_indices = {}   # {layer_name: faiss_index}
    grid_size = None

    with torch.inference_mode():
        start_time = time.time()

        for img_name in tqdm(img_ref_samples, desc="Extracting reference features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            aug_images = list(augment_image(img_rgb) if rotation else [img_rgb])
            ref_aug_mode = os.environ.get("SUPERAD_REF_AUG", "none").strip()
            if ref_aug_mode and ref_aug_mode != "none":
                aug_images.extend(augment_reference_illumination(
                    img_rgb, ref_aug_mode,
                    seed=abs(hash((img_name, ref_aug_mode, seed))) % (2**32)))
            for aug in aug_images:
                img_tensor, grid_size = model.prepare_image(aug)
                feats_dict = model.extract_features(img_tensor)
                # cls_feats_dict = model.extract_cls_features(img_tensor)
                idx=0
                for feats in feats_dict:
                    mask = model.compute_background_mask(feats, grid_size, threshold=1,
                                                         masking_type=(mask_ref_images and masking))
                    selected_feats = feats[mask]
                    if f'layer{idx}' not in feature_refs:
                        feature_refs[f'layer{idx}'] = []
                    feature_refs[f'layer{idx}'].append(selected_feats)
                    idx+=1

        # Concatenate and build FAISS index for each layer
        for layer_name, feats_list in feature_refs.items():
            layer_feats = np.concatenate(feats_list, axis=0).astype('float32')
            if knn_metric == 'L2_normalized':
                faiss.normalize_L2(layer_feats)

            if faiss_on_cpu:
                index = faiss.IndexFlatL2(layer_feats.shape[1])
            else:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, layer_feats.shape[1])


            index.add(layer_feats)
            knn_indices[layer_name] = index

        time_memorybank = time.time() - start_time

        inference_times = {}
        anomaly_scores = {}
        print(f"processing test samples ({object_name})")
        for test_type in ["test_private", "test_private_mixed"]:
            data_dir = f"{data_root}/{object_name}/{test_type}"

            os.makedirs(f"{plots_dir}/submission_folder/anomaly_images/{object_name}/{test_type}", exist_ok=True)
            os.makedirs(f"{plots_dir}/submission_folder/anomaly_images_thresholded/{object_name}/{test_type}", exist_ok=True)

            for idx, test_img_name in enumerate(sorted(os.listdir(data_dir))):
                start_time = time.time()
                test_path = f"{data_dir}/{test_img_name}"
                img_rgb = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)

                img_tensor, grid_size2 = model.prepare_image(img_rgb)
                feats_dict = model.extract_features(img_tensor)

                dists_per_layer = []

                mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1, masking_type=masking)
                for num, feats in enumerate(feats_dict):
                    masked_feats = feats[mask]

                    if knn_metric == "L2_normalized":
                        faiss.normalize_L2(masked_feats)

                    dists, _ = knn_indices[f'layer{num}'].search(masked_feats, k=knn_neighbors)
                    if knn_neighbors > 1:
                        dists = dists.mean(axis=1)

                    dists = 1 - dists  # cosine distance

                    dmap = np.zeros_like(mask, dtype=float)
                    dmap[mask] = dists.squeeze()
                    dmap = dmap.reshape(grid_size2)

                    dists_per_layer.append(dmap)

                # Average the 4 layers
                anomaly_map = np.mean(dists_per_layer, axis=0)
                score = mean_top1p(anomaly_map.flatten())

                inference_times[f"{test_img_name}"] = time.time() - start_time
                anomaly_scores[f"{test_img_name}"] = score

                test_img_name = test_img_name.split(".")[0]

                with open("./results_dir/metrics_seed=0.json", "r") as f:
                    object_thresholds = json.load(f)

                if save_tiffs:
                    anomaly_map_f16 = anomaly_map.astype(np.float16)
                    tiff.imwrite(
                        f"{plots_dir}/submission_folder/anomaly_images/{object_name}/{test_type}/{test_img_name}.tiff",
                        anomaly_map_f16
                    )

                    full_res_map = dists2map(anomaly_map, img_rgb.shape)
                    threshold = object_thresholds[object_name]["best_thre"]
                    binary_mask = (full_res_map > threshold).astype(np.uint8) * 255
                    
                    fill_config = {
                        'can': False,
                        'fabric': True,
                        'fruit_jelly': False,
                        'rice': False,
                        'sheet_metal': False,
                        'vial': False,
                        'wallplugs': False,
                        'walnuts': True
                    }
                    
                    if fill_config[object_name]:
                        binary_mask = fill_closed_regions(binary_mask)
                    cv2.imwrite(
                        f"{plots_dir}/submission_folder/anomaly_images_thresholded/{object_name}/{test_type}/{test_img_name}.png",
                        binary_mask
                    )


    return anomaly_scores, time_memorybank, inference_times
