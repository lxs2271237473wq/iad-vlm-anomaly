import cv2
import os
import sys
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np


# Base Wrapper Class
class VisionTransformerWrapper:
    def __init__(self, model_name, device, smaller_edge_size=224, half_precision=False):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")


# ViT-B/16 Wrapper
class ViTWrapper(VisionTransformerWrapper):
    def load_model(self):
        if self.model_name == "vit_b_16":
            model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
            self.transform = models.ViT_B_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_b_32":
            model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)
            self.transform = models.ViT_B_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        elif self.model_name == "vit_l_16":
            model = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
            self.transform = models.ViT_L_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_l_32":
            model = models.vit_l_32(weights = models.ViT_L_32_Weights.DEFAULT)
            self.transform = models.ViT_L_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        else:
            raise ValueError(f"Unknown ViT model name: {self.model_name}")
        
        model.eval()
        # print(self.transform)

        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor, self.grid_size

    def extract_features(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            patches = self.model._process_input(img_tensor)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches)
            return patch_features[:, 1:, :].squeeze().cpu().numpy()  # Exclude the class token

    def get_embedding_visualization(self, tokens, grid_size = (14,14), resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*self.grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False):
        # No masking for ViT supported at the moment... (Only DINOv2)
        return np.ones(img_features.shape[0], dtype=bool)
    

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):
        local_repo = os.environ.get("DINOV2_LOCAL_REPO")
        local_weights = os.environ.get("DINOV2_LOCAL_WEIGHTS")
        if local_repo and local_weights:
            repo_parent = os.path.dirname(os.path.abspath(local_repo))
            if repo_parent not in sys.path:
                sys.path.insert(0, repo_parent)
            from dinov2.hub import backbones as dino_backbones
            constructor = getattr(dino_backbones, self.model_name)
            model = constructor(pretrained=False)
            state = torch.load(local_weights, map_location="cpu")
            model.load_state_dict(state, strict=True)
            print(f"Loaded local DINOv2 weights: {local_weights}")
        else:
            model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)

        # Set transform for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        
        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size

    def set_smaller_edge_size(self, size):
        """Update both the public resolution value and the already-built resize transform."""
        self.smaller_edge_size = int(size)
        self.transform.transforms[0] = transforms.Resize(
            size=self.smaller_edge_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
    

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            # take a single layer
            # tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
            # take last 4 layers average
            # tokens = torch.stack(self.model.get_intermediate_layers(image_batch, 4)).mean(dim=0).squeeze()
            # take [5,11,17,23] layers average
            # tokens = torch.stack(self.model.get_intermediate_layers(image_batch, [5,11,17,23])).mean(dim=0).squeeze()
        
        # return tokens.cpu().numpy()
        
            layers = [5, 11, 17, 23]
            # The vendored official DINOv2 snapshot exposes the older API that
            # accepts only "last n layers". Reproduce the newer indexed API
            # explicitly and remove CLS/register tokens from patch features.
            tokens = self.model.prepare_tokens_with_masks(image_batch)
            tokens_list = []
            for layer_index, block in enumerate(self.model.blocks):
                tokens = block(tokens)
                if layer_index in layers:
                    normalized = self.model.norm(tokens)
                    start = 1 + int(getattr(self.model, "num_register_tokens", 0))
                    tokens_list.append(normalized[:, start:])
            features_list = [tokens.squeeze().cpu().numpy() for tokens in tokens_list]
        
        return features_list  # list of 4 arrays
    
    def extract_cls_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)
            tokens = self.model.prepare_tokens_with_masks(image_batch)
            for block in self.model.blocks:
                tokens = block(tokens)
            cls_token = self.model.norm(tokens)[:, 0].squeeze(0)
        return cls_token


    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens


    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)


    def compute_background_mask(self, img_features, grid_size, threshold = 1, masking_type = False, kernel_size = 3):
        # Output-identical fast path. With masking off, the value returned at the end of
        # this function is np.ones_like(first_pc).squeeze(), i.e. an all-True mask of
        # length n_tokens, which does not depend on the PCA at all. The randomized PCA
        # below costs several seconds per call and is discarded in that branch, so it is
        # skipped here. Results are bit-identical to the original implementation.
        if masking_type == False:
            return np.ones(img_features.shape[0], dtype=bool)
        # Kernel size for morphological operations should be odd
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        ############################################# Variance ################################################
        if masking_type == True:
            first_pc_flat = first_pc.squeeze() 
            mask = first_pc_flat > threshold 

            # Calculate the variance and adjust the mask
            mask_features = img_features[mask]
            non_mask_features = img_features[~mask]

            mask_variance = np.var(mask_features, axis=0) if mask_features.size > 0 else 0
            non_mask_variance = np.var(non_mask_features, axis=0) if non_mask_features.size > 0 else 0

            if np.median(mask_variance) < np.median(non_mask_variance):
                mask = ~mask
        
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()


def get_model(model_name, device, smaller_edge_size=448):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Smaller edge size: {smaller_edge_size}")

    if model_name.startswith("vit"):
        return ViTWrapper(model_name, device, smaller_edge_size)
    elif model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, device, smaller_edge_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
