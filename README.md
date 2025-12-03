# R2Gen 인코더 디코더 실험



<img width="1034" height="531" alt="image" src="https://github.com/user-attachments/assets/d3c89ed6-9693-452e-94a8-eb44bd1e24fe" />

## Installation

To get started, clone the repository and install dependencies:

```bash
!git clone https://github.com/zhjohnchan/R2Gen.git
%cd R2Gen
!ls
```

## Requirements

```bash
- nltk
- scikit-image
- matplotlib
- pycocotools
- pandas
- pillow
- tqdm
- numpy
- torch torchvision
- timm
```

## Datasets
We use datasets (IU X-Ray) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

## Download R2Gen Encoder (Encoder_experiment)
You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/1E44ufzy6K0IF3UQ0j6vAtydLCdtjnvqt?usp=drive_link).

## ⚠️ Custom File Replacement (중요)

본 프로젝트에서는 R2Gen 원본 코드에서 아래 두 파일을 커스텀 버전으로 수정하여 사용합니다.  
따라서 레포지토리를 클론한 뒤 **반드시 다음 두 파일을 교체해야 합니다.**

- `models/r2gen.py`
- `modules/visual_extractor.py`
---

## Run on IU X-Ray (Encoder_experiment)

To train the model on the IU X-Ray dataset, run the following command:

```bash
!python main_train.py \
  --dataset_name iu_xray \
  --ann_path "/content/iu_xray/annotation.json" \
  --image_dir "/content/iu_xray/images" \
  --save_dir "/content/iu_xray/saves/xcit_medium_24_p16_224" \
  --visual_extractor xcit_medium_24_p16_224 \
  --epochs 15
  --seed 9223
  # --visual_extractor resnet101
  # --visual_extractor densenet121
  # --visual_extractor vit_b_16
  # --visual_extractor vit_b_32
  # --visual_extractor xcit_small_12_p16_224
  # --visual_extractor xcit_medium_24_p16_224
```

## Test on IU X-Ray (Encoder_experiment)

To evaluate the trained model on the IU X-Ray dataset, run:

```bash
!python main_test.py \
  --dataset_name iu_xray \
  --ann_path "/content/iu_xray/annotation.json" \
  --image_dir "/content/iu_xray/images" \
  --save_dir "/content/iu_xray/saves/xcit_medium_24_p16_224" \
  --load "/content/iu_xray/saves/xcit_medium_24_p16_224/model_best.pth" \
  --visual_extractor xcit_medium_24_p16_224
```





