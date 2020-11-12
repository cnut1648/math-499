# MATH499 Final Project Team2

![Project Image](https://www.researchgate.net/profile/Gearoid_OLaighin/publication/8567542/figure/fig1/AS:667210611691528@1536086814564/Discriminating-postures-a-standing-b-sitting-c-lying-The-arrows-indicate-the.png)

> This is a final deliverable for LookDeep Inc that determines huaman posture in a medical setting through convolution.

---

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description

This project trains on convolution networks with over ~40000 medical images provided by LookDeep Inc. The goal of this project is to determine whether the person of interest is either 1.Standin, 2.Sitting, or 3.Lying. 

#### Archetectures

- MobileNet 
- EfficientNetB0 
- Team2Net



---

## How To Use

The execution of this code requires the change of label_dir,img_dir, and save_dir in config.py.
- label_dir: the directory that contains the labeling csv files
- img_dir: the directory that contains all of the testing images, where images from tranch t must be in subfolder img_dir/tranch<t> and images for all tranchs must be in subfolder img_dir/allTranch.
- save_dir: the directory used to save model checkpoints in save_dir/<tranch>/<model type>-<ensemble num>.<epoch>-<val acc>.h5
  
Then, to execute, run train.py.

#### Installation

Required Packages: 
- `tensorflow 2.3` 

- `albumentation`

- [`imagedataaugmenter`](https://github.com/mjkvaak/ImageDataAugmentor) 

  Fork of Keras image data generator which supports the 3rd party data augmentation modules
  
  ---

## References

- Buslaev, A., Parinov, A., Khvedchenya, E., Iglovikov, V., & Kalinin, A. (2018, September 18). Albumentations: Fast and flexible image augmentations. Retrieved November 12, 2020, from https://arxiv.org/abs/1809.06839

- He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10). Deep Residual Learning for Image Recognition. Retrieved November 12, 2020, from https://arxiv.org/abs/1512.03385

- Huang, G., Liu, Z., Van der Maaten, L., & Weinberger, K. (2018, January 28). Densely Connected Convolutional Networks. Retrieved November 12, 2020, from https://arxiv.org/abs/1608.06993

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. (2019, March 21). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Retrieved November 12, 2020, from https://arxiv.org/abs/1801.04381

- Tan, M., & Le, Q. (2020, September 11). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Retrieved November 12, 2020, from https://arxiv.org/abs/1905.11946

- Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017, November 16). Random Erasing Data Augmentation. Retrieved November 12, 2020, from https://arxiv.org/abs/1708.04896

---

## License

MIT License

Copyright (c) [2017] [James Q Quick]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[Back To The Top](#MATH499-Final-Project-Team2)

---

## Contributors

- Daniel Chaderjian, Jiashu Xu, Phillip Bliss, Sydney Yu, Terry Lu

[Back To The Top](#MATH499-Final-Project-Team2)
