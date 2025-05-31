import torch
import cv2
import numpy as np
from app.models.u2net import U2NET

class U2NetWrapper:
    def __init__(self, weights_path: str, device: str = "cuda"):
        # Load cuda version of torch
        self.device = torch.device(device)
        self.model = U2NET(3, 1).to(self.device)
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
        # 1) BGR→RGB, resize if desired
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = rgb.shape

        # 2) normalize to [0,1], to torch tensor
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2,0,1)
        tensor = tensor.unsqueeze(0).to(self.device)

        # 3) forward
        with torch.no_grad():
            d1, *_ = self.model(tensor)     # U²-Net returns multiple side-outputs; d1 is final
        mask = torch.sigmoid(d1)[0,0]       # shape H×W

        # 4) back to numpy, resize to original
        sal = mask.cpu().numpy()
        sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)

        # 5) normalize again just in case
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
        return True, sal.astype(np.float32)