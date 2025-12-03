# R2Gen 인코더 디코더 실험



<img width="1323" height="648" alt="image" src="https://github.com/user-attachments/assets/51b85df0-dc4a-4eb4-a9ee-da989556447a" />

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
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

## Download R2Gen Encoder 
You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/1E44ufzy6K0IF3UQ0j6vAtydLCdtjnvqt?usp=drive_link).

---

## 📘 Run on IU X-Ray

To train the model on the IU X-Ray dataset, run the following command:

```bash
!python main_train.py \
  --image_dir /content/iu_xray/images \
  --ann_path /content/iu_xray/annotation.json \
  --dataset_name iu_xray \
  --max_seq_length 60 \
  --threshold 3 \
  --batch_size 16 \
  --epochs 15 \
  --save_dir "/content/drive/MyDrive/Colab Notebooks/APID" \
  --step_size 50 \
  --gamma 0.1 \
  --seed 9223
```

## 📘 Test on IU X-Ray

To evaluate the trained model on the IU X-Ray dataset, run:

```bash
!python main_test.py \
  --image_dir /content/iu_xray/images \
  --ann_path /content/iu_xray/annotation.json \
  --dataset_name iu_xray \
  --max_seq_length 60 \
  --threshold 3 \
  --batch_size 16 \
  --save_dir "/content/resnet101" \
  --seed 9223 \
  --load "/content/iu_xray/saves/resnet101/model_best.pth"
```





