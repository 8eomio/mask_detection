# Who is Good Masker

![0bc8d1b7-4134-411c-9094-edcde243c053](https://github.com/8eomio/mask_detection/assets/61742009/fa69dba0-dc9a-4134-a371-f1200e554913)

## Who is Good Masker: Discriminate whether people wear masks well
**Objective**: Develop a real-time mask detection system using CNN to monitor mask compliance in public spaces.

**2-stage Detection:**
Face Detection: Extract facial regions using an open-source face detector library.
Mask Classification: Classify extracted facial regions into four categories (no mask, chin mask, nose mask, valved mask) using CNN-based transfer learning with various backbone models (SqueezeNet, AlexNet, ResNet50, MobileNet, VGG).

## Mask Wearing Status Feedback

| Mask Wearing Status | Feedback Message |
|---|---|
| Good Masker! | Mask worn correctly. |
| No mask..Please wear a mask!. | Please wear a mask. |
| Valve Mask, please wear the proper types of mask | Please wear a mask that complies with防疫 regulations. |
| Chin Mask! put the mask up to your nose | Please pull your mask up to cover your nose. |
| Nose mask! put the mask up to your nose | Please pull your mask up to cover your nose. |

## Dataset
The training dataset is availble from [dataset link](https://drive.google.com/drive/folders/1tnG8xWhuBPvWTOgKGUsQ6XFwjwarJsv1?usp=sharing)
The dataset consists of **5 classes**: Wearing Masks Well, Not wearing masks, Wearing masks under the chin, Covering only the nose, Valve masks

The initial dataset for this project was obtained from Kaggle. However, the images found on Kaggle were limited to two classes: individuals wearing masks correctly and individuals not wearing masks at all. To address this limitation, images of individuals wearing masks under the chin (chin mask), covering only the nose (nose mask), or using valve masks were created by synthesizing mask images with human face images. However, time constraints prevented the generation of a large dataset through image synthesis alone. Therefore, augmentation techniques were employed to increase the number of images using OpenCV library functions to rotate images at different angles, flip them horizontally, and apply other transformations.


## requirement
pip install opencv-python
pip install matplotlib
pip install torchvision
pip install cvlib
pip install tensorflow
