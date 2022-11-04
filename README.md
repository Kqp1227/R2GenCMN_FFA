# R2GenCMN_FFA-IR

The framework is inherited from [R2GenCMN](https://github.com/zhjohnchan/R2GenCMN).

## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Datasets
(Z. Chen et al. 2021) use two datasets (IU X-Ray and MIMIC-CXR) in their paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/) and then put the files in `data/mimic_cxr`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.
There have been some works using `MIMIC-CXR` only and treating the whole `IU X-Ray` dataset as an extra test set.



I use `FFA-IR` dataset from [FFA-IR](https://github.com/mlii0117/FFA-IR), including all FFA images and annotation files, is available on [PhysioNet](https://physionet.org/content/ffa-ir-medical-report/1.0.0/). 

## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

Run `bash train_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

Run `bash train_ffa.sh` to train a model on the FFA-IR data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.

## Visualization

Run `bash plot_mimic_cxr.sh` to visualize the attention maps on the MIMIC-CXR data.
