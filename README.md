## Credits to Model Repositories
1. XueBinQin & LeCongThuong's U^2-Net:
https://github.com/LeCongThuong/U2Net
2. Yun Liu's FastSaliency (SAM-Net):
https://github.com/yun-liu/FastSaliency
3. Matthias Kümmerer's Pysaliency:
https://github.com/matthias-k/pysaliency

## Dataset Link
https://cocodataset.org/#download

## Installation Tutorial 
1. Install Anaconda & add to path
2. Install Octave & Strawberry PERL & add to path
* choco install strawberryperl on Windows
* This software is only required for AIM & SUN models
3. cd into source code folder
4. conda env create -f environment.yml
5. Move downloaded COCO_val2017 images into data/COCO
6. Move u2.pth file into models/u2net
7. Move SAMNet_with_ImageNet_pretrain.pth into models/samnet
8. Run segmentation stats with python -m app.stats.seg_stats