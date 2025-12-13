# 4EU+ DeepLife Deep Learning in Life Sciences - Image Segmentation Project

Deep Learning models for Multi-modal Microscopy Image Segmentation

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F37626?style=for-the-badge&logo=google&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C2D91?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003366?style=for-the-badge&logo=plotly&logoColor=white)

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/img/DL_logo_header.png" alt="DL_logo">
</p>

The project is based on a 2022 [Cell Segmentation challenge proposed at NeurIPS](https://neurips22-cellseg.grand-challenge.org/) containing multi-modal microscopy images. The competition proceeding have been additionally published on [PMLR](https://proceedings.mlr.press/v212/). The dataset itself is accessible in [CellSeg](https://neurips22-cellseg.grand-challenge.org/dataset/).

Project done under the supervision of **prof. Elena Casiraghi** (University of Milan) with teams from **Sorbonne University**, **University of Milan**, and **University of Warsaw**. The project culminated with a DeepLife Hackathon in **Heidelberg University** in June 2025.

Team Warsaw:
- [Younginn Park](https://github.com/young-sudo)
- [Mateusz Chojnacki](https://github.com/M-Chojnacki6)
- [Lidia Stadnik](https://github.com/lidst)
- Ignacy Makowski

# Methods

Our team focused on 4 model architectures for this task: U-Net, U-Net++, PSPNet, and Swin-UNet.

## U-Net

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/img/unet.png" alt="unet">
</p>

## U-Net++

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/img/unetpp.png" alt="unetpp">
</p>

## PSPNet

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/img/pspnet.png" alt="pspnet">
</p>

## Swin-UNet

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/img/swinunet.png" alt="swinunet">
</p>

# Results

Table. Evaluation metrics across all of our models. (A - Data augmentation, N - Batch normalization)
| Model  | Dice  | IoU  |
|---|---|---|
| U-Net++  | 0.77  | 0.62   |
| SwinUNet  | 0.85  | 0.77   |
| PSPNet | 0.75  | 0.60   |
| U-net (1 class) | 0.74 | 0.59 |
| U-net (1 class)+A | 0.82 | 0.70 | 
| U-net (2 class)+A | 0.94 | 0.89 |
| U-net (2 class)+A+N | 0.92 | 0.86 |

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/img/metrics.png" alt="pspnet" width="200">
    <br>
    <small>Evaluation metrics</small>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/results/UNet_sample_predictions.png" alt="UNet_preds" width="300">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/results/UNet_threshold_curves.png" alt="UNet_curve" width="300">
    <br>
    <small>Segmentation results and threshold curve for U-Net with Data Augmentation</small>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/results/UNet++_sample_predictions.png" alt="UNetpp_preds" width="300">
    <img src="https://raw.githubusercontent.com/young-sudo/imseg-net/main/results/UNet++_threshold_curves.png" alt="UNetpp_curve" width="300">
    <br>
    <small>Segmentation results and threshold curve for U-Net++</small>
</p>

# Acknowledgments

Team Milan 
- Emad Bahreinipour
- Arash Khosropour

Team Sorbonne
- Philys Matsimouna
- Estela Stankov
- Tess Chilliet
- Anabelle Joubin

Team Sorbonne 2
- Aliénor Collas
- Mathis De Jesus
- Ernestine Félicien
- Gabriel Pelle-Huet

# References

Dataset - Ma, J., Xie, R., Ayyadhury, S., Ge, C., Gupta, A., Gupta, R., Gu, S., Zhang, Y., Lee, G., Kim, J., Lou, W., Li, H., Upschulte, E., Dickscheid, T., de Almeida, J. G., Wang, Y., Han, L., Yang, X., Labagnara, M., … Wang, B. (2024). NeurIPS 2022 Cell Segmentation Competition Dataset [Data set]. Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), New Orleans. Zenodo. https://doi.org/10.5281/zenodo.10719375

U-Net - O. Ronneberger and P.Fischer and T. Brox,  "U-Net: Convolutional Networks for Biomedical Image Segmentation", Medical Image Computing and Computer-Assisted Intervention (MICCAI), LNCS, 9351, 234--241, 2015, Springer, http://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a

U-Net++ - Zongwei Zhou and Md Mahfuzur Rahman Siddiquee and Nima Tajbakhsh and Jianming Liang, UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation, 2020,  1912.05074, hhttps://arxiv.org/abs/1912.05074

PSPNet - Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya, Pyramid Scene Parsing Network,  CVPR, 2017

Swin U-Net - Hatamizadeh et al., 2022. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation

IoU - Padilla, Rafael & Netto, Sergio & da Silva, Eduardo. (2020). A Survey on Performance Metrics for Object-Detection Algorithms. 10.1109/IWSSIP48289.2020.

Dice - Rao, Divya & K, Prakashini & Singh, Rohit & J, Vijayananda. (2022). Automated segmentation of the larynx on computed tomography images: a review. Biomedical Engineering Letters. 12. 1-9. 10.1007/s13534-022-00221-3.

Segmentation Models in PyTorch - Iakubovskii, Pavel. (2019). Segmentation Models Pytorch. GitHub repository. GitHub. https://github.com/qubvel/segmentation_models.pytorch
