# ECCV2020-OSAD

Pytorch codes for Open-set Adversarial Defense <a href=https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620664.pdf> (pdf) </a> in ECCV 2020 

Network structure of the proposed Open-Set Defense Network (OSDN). It consists of four components: encoder, decoder, open-set classiﬁer and transformation
classiﬁer.

<img src="./models/framework.png" width="500">

Overview of proposed framework. We simulate domain shift by randomly dividing original N source domains in each iteration. Supervision of domain knowledge is incorporated via depth estimator to regularize the learning process of feature extractor. Thus, meta learner conducts the meta-learning in the feature space regularized by the auxiliary supervision of domain knowledge. 

<img src="./models/framework.png" width="600">

# Setup

* Prerequisites: Python3.6, pytorch=0.4.0, Numpy, TensorboardX, Pillow, SciPy, h5py

* The source code folders:

  1. "models": Contains the network architectures suitable for high-order derivatives calculation of network parameters. Please note that FeatExtractor, DepthEstmator and FeatEmbedder in the code are feature extractor, depth estimator and meta learner in the paper, respectively. 
  2. "core": Contains the training and testing files. Note that we generate score for each frame during the testing.
  3. "datasets": Contains datasets loading
  4. "misc": Contains initialization and some preprocessing functions
  
# Training

To run the main file: python main.py --training_type Train

# Testing

To run the main file: python main.py --training_type Test

It will generate a .h5 file that contains the score for each frame. Then, we use these scores to calculate the AUC and HTER.

# Acknowledge
Please kindly cite this paper in your publications if it helps your research:
```
@InProceedings{Shao_2020_AAAI,
author = {Shao, Rui and Lan, Xiangyuan and Yuen, Pong C.},
title = {Regularized Fine-grained Meta Face Anti-spoofing},
booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
year = {2020}
}
```

Contact: ruishao@comp.hkbu.edu.hk
