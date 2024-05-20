# Machine-Learning-Finals-Project

## Project Description

Cerebral microbleeds (CMBs) are small, chronic brain hemorrhages detectable via MRI scans due to their paramagnetic properties. Accurately detecting these microbleeds is challenging due to their resemblance to other brain artifacts, leading to high rates of false positives and variability in detection outcomes among radiologists. Recent studies have explored various deep learning approaches to enhance the detection of cerebral microbleeds.

This project builds on the work of Lee et al. (2022) and their Single-Stage Triplanar Ensemble Detection Network (TPE-Det). Our approach modifies this existing model to focus specifically on axial plane images from MRI scans using the VALDO dataset (Sudre et al., 2024) . The goal is to improve the precision, sensitivity, and reduce the average number of false positives in the detection of cerebral microbleeds.

## Setup Instructions

#### 1. Clone the Repository

#### 2. Install Required Packages

```
pip install -r requirements.txt
```

#### 3. Download and Setup the VALDO Dataset

1. Go here and follow the instructions on what you need to do to download and use their provided dataset (https://valdo.grand-challenge.org/).
2. Place the dataset into the ../VALDODataset folder relative to this project directory.

#### 4. Download and Prepare the Pretrained Weights

1. Go to: https://github.com/rwightman/efficientdet-pytorch/releases.
2. Look for weights in the bottom.
3. Find `tf_efficientdet_d7-f05bf714.pth` and download it.
4. Move the downloaded file into the root of this project repository.
5. Rename the file to `efficientdet_d7-f05bf714.pth`

## Additional Notes

- Always ensure you comply with data use agreements and legal restrictions for datasets like VALDO.

## References

- Lee, H., Kim, J., Lee, S., Jung, K., Kim, W., Noh, Y., Kim, E. Y., Kang, K. M., Sohn, C., Lee, D. Y., Al‐masni, M. A., & Kim, D. (2023). Detection of cerebral microbleeds in mr images using a single‐stage triplanar ensemble detection network(Tpe‐det). Journal of Magnetic Resonance Imaging, 58(1), 272–283. https://doi.org/10.1002/jmri.28487
- Sudre, C. H., Van Wijnen, K., Dubost, F., Adams, H., Atkinson, D., Barkhof, F., Birhanu, M. A., Bron, E. E., Camarasa, R., Chaturvedi, N., Chen, Y., Chen, Z., Chen, S., Dou, Q., Evans, T., Ezhov, I., Gao, H., Girones Sanguesa, M., Gispert, J. D..., de Bruijne, M. (2024). Where is VALDO? Vascular lesions detection and segmentation challenge at miccai 2021. Medical Image Analysis, 91, 103029. https://doi.org/10.1016/j.media.2023.103029
