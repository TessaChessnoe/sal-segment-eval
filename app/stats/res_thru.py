def measure_resolution_throughput(detectors: dict, dataset: list, sample_n: int = 50):
    import time
    from tqdm import tqdm
    from random import sample

    results = []

    for name, detector in detectors.items():
        print(f"→ Measuring resolution throughput for {name}")
        samples = sample(dataset, sample_n)

        total_pixels = 0
        total_time = 0.0
        successful = 0

        for fn, img, gt_mask in tqdm(samples, desc=name):
            h, w = img.shape[:2]
            n_pixels = h * w

            start = time.time()
            if hasattr(detector, "computeSaliency"):
                success, _ = detector.computeSaliency(img)
            else:
                sal_map = detector.saliency_map(img)
                success = sal_map is not None
            end = time.time()

            if success:
                total_pixels += n_pixels
                total_time += (end - start)
                successful += 1

        if successful == 0:
            print(f"⚠️  {name} failed all images!")
            continue

        throughput = total_pixels / total_time  # pixels per second
        mp_throughput = throughput / 1_000_000

        print(f"✓ {name}: {mp_throughput:.2f} MP/s")

        results.append({
            "Model": name,
            "Pixels Processed": total_pixels,
            "Time (s)": total_time,
            "MP/s": mp_throughput,
            "Samples Used": successful
        })

    return results

# Paths for pretrained weights & model files
PYSAL_ROOT = "app/models/pysal"
U2_WEIGHTS = "app/models/u2net/u2net.pth"
U2P_WEIGHTS = "app/models/u2net/u2netp.pth"
SAM_WEIGHTS = "app/models/samnet/SAMNet_with_ImageNet_pretrain.pth"

# Paths for image files and annotation json
ANN_PATH = "data/COCO/annotations/instances_val2017.json"
INP_IMG_DIR = "data/COCO/val2017"

if __name__ == "__main__":
    import pandas as pd
    import cv2
    from app.stats.stat_helpers import gather_dataset

    # Import custom models & wrappers
    # import pysaliency as pys
    from app.models.custom_models import BMSOptimized, IttiKoch
    from app.models.u2net.u2_wrapper import U2NetWrapper, U2NetPWrapper
    from app.models.samnet.samnet_wrapper import SAMNetWrapper

    DETECTORS = {
        "U2Net": U2NetWrapper(weights_path=U2_WEIGHTS),
        "U2NetP": U2NetPWrapper(weights_path=U2P_WEIGHTS),
        "SAMNet": SAMNetWrapper(weights_path=SAM_WEIGHTS),
        # "AIM": pys.AIM(location=PYSAL_ROOT),
        # "SUN": pys.SUN(location=PYSAL_ROOT),
        "Finegrain": cv2.saliency.StaticSaliencyFineGrained.create(),
        "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual.create(),
        "BMS": BMSOptimized,
        "IKN": IttiKoch,
    }
    dataset = gather_dataset(ANN_PATH, INP_IMG_DIR)
    results = measure_resolution_throughput(DETECTORS, dataset, sample_n=500)
    df = pd.DataFrame(results)
    df = df.sort_values(by="MP/s", ascending=False)
    df.to_csv("res_thru.csv", index=False)
    print(df)


