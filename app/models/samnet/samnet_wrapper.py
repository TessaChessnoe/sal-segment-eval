import torch 
import cv2
import numpy as np
from app.models.samnet.SAMNet import FastSal
import torch.nn.functional as F
import os

class SAMNetWrapper:
    def __init__(self, device: str="cuda", weights_path = None):
        # 1) Instiantiate model using cpu/gpu 
        self.device = torch.device(device)
        self.model = FastSal()
        # If no explicit weights_path given, default to the “official” filename
        if weights_path is None:
            # __file__ is ".../app/models/samnet/samnet_wrapper.py"
            base_dir = os.path.dirname(__file__)
            weights_path = os.path.join(
                base_dir, "SAMNet_with_ImageNet_pretrain.pth"
            )
        raw_state = torch.load(
            weights_path, 
            map_location = self.device,
            weights_only = True
        )

        # print("→ [DEBUG] torch.load returned object of type:", type(raw_state))

        # # If it’s a dict, print its top‐level keys:
        # if isinstance(raw_state, dict):
        #     print("→ [DEBUG] Top‐level keys in checkpoint:", list(raw_state.keys())[:10])
        # else:
        #     print("→ [DEBUG] Loaded object is not a dict:", raw_state)

        # 2) Strip module header from weights dictionary
        state = {}
        l = len("module.")
        for k,v in raw_state.items():
            if k.startswith("module."):
                k_strip = k[l:]
                state[k_strip] = v
            else:
                state[k] = v

        # 3) Load model & weights for evaluation
        # Load weights dict into model
        self.model.load_state_dict(state)
        # Load model onto torch device
        self.model.to(self.device)
        # Enable evaluation mode
        self.model.eval()
    
    def preprocess_input(self, img: np.ndarray) -> torch.Tensor:
        # Convert BGR (as loaded by OpenCV) to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # H × W × 3 → 3 × H × W
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor  # shape: (1, 3, H, W)
    
    def postprocess_output(self, sal_map: torch.Tensor,
                           target_size: tuple[int, int]) -> np.ndarray:
        """
        Given the raw model output (a torch.Tensor of shape [1, 1, h', w']),
        interpolate it back to the original image’s size (H, W), then convert
        to a float NumPy array in [0, 1].
        """
        # Suppose sal_map is shape (1, 1, h', w'). Upsample to original (H, W):
        upsampled = F.interpolate(
            sal_map, size=target_size, mode="bilinear", align_corners=False
        )
        arr = upsampled.squeeze().detach().cpu().numpy()
        # Ensure values are in [0, 1]:
        arr = np.clip(arr, 0.0, 1.0)
        return arr  # shape: (H, W), float32
    
    def computeSaliency(self, img: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """
        Compute SAMNet saliency map for a single image.
        Returns (success: bool, saliency_map: np.ndarray).
        - success=False if something went wrong (e.g. all zeros).
        - saliency_map is a float32 map in [0, 1], shape = (H, W).
        """
        # 1) Preprocess
        h, w = img.shape[:2]
        inp = SAMNetWrapper.preprocess_input(self, img).to(self.device)

        # 2) Forward pass (no grad)
        with torch.no_grad():
            out = self.model(inp)
            # 0th index of model output is the logits
            if isinstance(out, (list, tuple)):
                sal_logits = out[0]
            else:
                sal_logits = out

        # 3) Postprocess: resize to original dims
        sal = SAMNetWrapper.postprocess_output(self, sal_logits, (h, w))

        # 4) Check if the result is non-trivial (e.g. not all zeros)
        if np.allclose(sal, 0.0):
            return False, None
        return True, sal
    