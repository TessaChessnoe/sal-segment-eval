import numpy as np
import cv2
from skimage.color import rgb2gray

class IttiKoch:
    @staticmethod
    def _normalize_map(m):
        """Normalize so that peaks stand out: 
           divide by (max+ε), then multiply by (1 – mean)."""
        m = np.abs(m)
        M = m.max()
        if M == 0:
            return np.zeros_like(m)
        m = m / M
        return m * (1 - m.mean())

    @staticmethod
    def _center_surround(feature, c=3, s=9):
        """DoG: difference of Gaussian-blurred maps"""
        small = cv2.GaussianBlur(feature, (0,0), c)
        large = cv2.GaussianBlur(feature, (0,0), s)
        return large - small

    @staticmethod
    def computeSaliency(img, resize_to=None):
        """
        img: HxWx3 BGR or RGB image
        returns: success flag, saliency map in [0,1]
        """
        if img is None:
            return False, None

        # 1) convert & optionally resize
        if img.ndim == 3:
            imgf = img.astype(np.float32) / 255.0
        else:
            imgf = np.dstack([img]*3).astype(np.float32) / 255.0

        if resize_to is not None:
            imgf = cv2.resize(imgf, resize_to, interpolation=cv2.INTER_AREA)

        H, W, _ = imgf.shape

        # 2) Intensity channel
        I = cv2.cvtColor((imgf*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

        # 3) Color channels
        R, G, B = imgf[:,:,2], imgf[:,:,1], imgf[:,:,0]
        rg = R - G
        by = B - (R+G)/2

        # 4) Orientation channels
        angles = [0, 45, 90, 135]
        O = []
        for angle in angles:
            # rotate image so that sobel-x gives response at desired orientation
            M = cv2.getRotationMatrix2D((W/2,H/2), angle, 1)
            rot = cv2.warpAffine(I, M, (W,H), flags=cv2.INTER_LINEAR)
            sob = cv2.Sobel(rot, cv2.CV_32F, 1, 0, ksize=7)
            # rotate back
            iM = cv2.invertAffineTransform(M)
            sob = cv2.warpAffine(sob, iM, (W,H), flags=cv2.INTER_LINEAR)
            O.append(sob)

        # 5) Build feature maps with center-surround
        feats = []
        for feat in [I, rg, by] + O:
            cs = IttiKoch._center_surround(feat)
            feats.append(IttiKoch._normalize_map(cs))

        # 6) Conspicuity maps: sum intensity, sum color, sum orientation
        CI = feats[0]
        CC = feats[1] + feats[2]
        CO = sum(feats[3:])

        # 7) Final map: normalize & add
        sal = np.stack([CI, CC, CO], axis=0).sum(0)
        if sal.max() > sal.min():
            sal = (sal - sal.min()) / (sal.max() - sal.min())
            return True, sal.astype(np.float32)
        else:
            return False, None

class BMS:
    @staticmethod
    def binarize_img(gray, threshold):
        # Use faster numpy vector comparison to threshold image
        return (gray > threshold).astype(np.uint8)

    @staticmethod
    def activate_bool_map(bool_map):
        """
        Label connected components, then keep only
        those that do NOT touch the image border.
        Gestalt principle: areas surrounded with contrasting info are salient. 
        """
        # connectedComponentsWithStats gives:
        #   num_labels, label_map, stats, centroids
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(bool_map, connectivity=8)
        h, w = bool_map.shape
        attn = np.zeros_like(bool_map, dtype=np.float32)

        # stats[i] = [x, y, width, height, area]
        for label in range(1, num_labels):  # skip background=0
            x, y, width, height, area = stats[label]
            # If region touches any border, 
            # its bbox will start at 0 
            # or extend up to w/h
            if x == 0 or y == 0 or x + width >= w or y + height >= h:
                continue
            # Otherwise activate those pixels
            attn[label_map == label] = 1.0

        return attn

    @staticmethod
    def computeSaliency(img, n_thresholds=16, lb=25, ub=230):
        """
        img: BGR or RGB image array (HxWx3)
        returns: success flag, saliency map normalized to [0,1]
        """
        # 0) Validate input
        if img is None:
            print("Image not found.")
            return False, None
        if img.ndim not in [2,3]:
            print(f"Invalid image dim: {img.ndim} Must be 2 or 3")
            return False, None
        # 1) Convert to gray float in [0,255]
        gray = rgb2gray(img) * 255.0
        # 2) Generate thresholds (equally spaced)
        thresholds = np.linspace(lb, ub, n_thresholds, endpoint=False)

        # 3) For each threshold, build and activate the boolean map
        attn_map = np.zeros(gray.shape, dtype=np.float32)
        for thr in thresholds:
            bool_map = BMS.binarize_img(gray, thr)
            attn_map += BMS.activate_bool_map(bool_map)

        # 4) Smooth the attention map
        attn_map = cv2.GaussianBlur(attn_map, ksize=(0, 0), sigmaX=3)

        # 5) Normalize to [0,1]
        if attn_map.max() > attn_map.min(): # Conditoinal prevents 0 div when max=min
            sal_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        # If max=min, bool maps likely were all 0
        # or dropped (all vals touched borders)
        else:
            return False, None
        return True, sal_map

class BMSFast:
    """
    Boolean Map Saliency (Fast) implementation.
    """
    def __init__(self):
        pass

    @staticmethod
    def _activate_bool_map(bool_map: np.ndarray) -> np.ndarray:
        """
        Given a binary map (uint8, values 0 or 1), returns a float32
        map of 1.0 in those connected components that do NOT touch the border.
        Uses floodFill mask to isolate interior blobs.
        """
        # ensure uint8 in 0/255 for floodFill
        im = (bool_map * 255).astype(np.uint8)
        h, w = im.shape

        # mask for floodFill must be 2 pixels larger
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)

        # flood fill from top-left corner to mark border-connected background in mask
        flags = cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(
            im,               # input image (unused output here)
            ff_mask,          # mask output
            (0, 0),           # seed point
            (1,),             # newVal ignored with MASK_ONLY, but must be Scalar
            loDiff=(0,),
            upDiff=(0,),
            flags=flags
        )

        # extract filled mask (remove padding): 1 marks background, 0 marks interior+foreground
        mask = ff_mask[1:-1, 1:-1]

        # interior blobs are where original foreground (bool_map==1) and mask==0
        interior = (bool_map == 1) & (mask == 0)
        return interior.astype(np.float32)

    @staticmethod
    def computeSaliency(img: np.ndarray,
                        n_thresholds: int = 16,
                        lb: int = 25,
                        ub: int = 230) -> np.ndarray:
        """
        Compute fast BMS saliency map.
        Returns saliency map (float32 [0,1]).
        """
        if img is None:
            raise ValueError("Input image is None")
        # convert to grayscale uint8
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img.copy()
        else:
            raise ValueError(f"Unsupported image dimension: {img.ndim}")

        # define thresholds
        thresholds = np.linspace(lb, ub, n_thresholds, endpoint=False).astype(np.uint8)

        # accumulate interior detections
        acc = np.zeros_like(gray, dtype=np.float32)
        for thr in thresholds:
            _, bm = cv2.threshold(gray, int(thr), 1, cv2.THRESH_BINARY)
            interior = BMSFast._activate_bool_map(bm)
            acc += interior

        # smooth and normalize
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=3)
        minv, maxv = float(acc.min()), float(acc.max())
        if maxv <= minv:
            raise RuntimeError("Saliency map has zero dynamic range; check input or thresholds.")
        sal = (acc - minv) / (maxv - minv)
        return sal.astype(np.float32)
