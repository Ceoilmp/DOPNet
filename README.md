# DOPNet: Dense Object Prediction Network for Multi-Class Object Counting and Localization in Remote Sensing Images

Code DOPNet: Dense Object Prediction Network for Multi-Class Object Counting and Localization in Remote Sensing Images.

Pre-trained models
---

[Baidu Cloud](链接：https://pan.baidu.com/s/1qOP0S6kLz5F3eBPS2zz1Hg) : n551

Environment
---
We are good in the environment:

python 3.8

pytorch 1.10.0

numpy 1.21.4

matplotlib 3.6.0

mmcv-full 1.4.8

Usage
---
We provide the test code for our model. 
The `DOPNet_RSOC.pth` model is adapted on the RSOC dataset. 
We randomly select an image from the RSOC_small-vehicle dataset and place it in the image folder.
And you can either choose the other images for a test.

We are good to run:

```
python test.py --model_state ./weights/DOPNet_RSOC.pth --out ./out/result.png
```

We will release more trained models soon.
The core code will be released after the journal paper is accepted.
Please see the paper for more details.

Acknowledgement
---

Thanks to these repositories
- [C-3 Framework](https://github.com/gjy3035/C-3-Framework)
- [mmcv](https://github.com/open-mmlab/mmcv)

If you have any question, please feel free to contact us. (ceoilmp@whu.edu.cn and gcding@whu.edu.cn)
