import torch
import cv2
import numpy as np
import os
from app.models.u2net.u2net import U2NET, U2NETP

class U2NetWrapper:
    def __init__(self, weights_path = None, device: str = "cuda"):
        # Load cuda version of torch
        self.device = torch.device(device)
        # Instantiate model
        self.model = U2NET(3, 1)
        # Load model onto GPU
        self.model.to(self.device)

        # If no explicit weights_path given, default to the “official” filename
        if weights_path is None:
            base_dir = os.path.dirname(__file__)
            weights_path = os.path.join(base_dir, "u2net.pth")

        # Load model state
        state = torch.load(weights_path,
                            map_location = self.device, 
                            weights_only = True)
        self.model.load_state_dict(state)
        # Enable evaluation
        self.model.eval()

    def computeSaliency(self, img: np.ndarray):
        """
        img: HxWx3 BGR array (0–255 uint8)
        returns: (True, sal_map) where sal_map is float32 [0,1] HxW
        """
        # 1) BGR->RGB, resize if desired
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = rgb.shape

        # 2) normalize to [0,1], to torch tensor
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2,0,1)
        tensor = tensor.unsqueeze(0).to(self.device)

        # 3) forward through U2NET
        with torch.no_grad():
            d1, *_ = self.model(tensor)     # U²-Net returns multiple side-outputs; d1 is final
        mask = torch.sigmoid(d1)[0,0]       # shape H×W

        # 4) back to numpy, resize to original
        sal = mask.cpu().numpy()
        sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)

        # 5) normalize again just in case
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
        return True, sal.astype(np.float32)

class U2NetPWrapper:
    def __init__(self, weights_path = None, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = U2NETP(3, 1)
        self.model.to(self.device)

        if weights_path is None:
            base_dir = os.path.dirname(__file__)
            weights_path = os.path.join(base_dir, "u2net.pth")

        state = torch.load(weights_path,
                            map_location = self.device, 
                            weights_only = True)
        self.model.load_state_dict(state)
        self.model.eval()
    def computeSaliency(self, img: np.ndarray):
        import cv2
        import numpy as np
        import torch.nn.functional as F

        # 1) BGR→RGB, normalize, to (1×3×H×W) float tensor on GPU
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W, _ = rgb.shape
        tensor = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(self.device)

        # 2) Forward through U2NETP (returns d0,d1,…,d6); take d0 as final
        with torch.no_grad():
            d0, *_ = self.model(tensor)
        mask = torch.sigmoid(d0)[0,0]  # shape (h', w')

        # 3) Upsample to original resolution
        sal = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False).squeeze().cpu().numpy()

        # 4) Normalize to [0,1]
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
        return True, sal.astype(np.float32)
        