# Visual identification of cattle from depth imagery

This repository contains the code accompanying my thesis "Recognition of Individual Cattle from Depth Imagery Using Deep Metric Learning". 

The implemention is built directly upon the framework given alongside Andrew, W. et al's work, "Visual Identification of Individual Holstein-Friesian Cattle via Deep Metric Learning", published in 2020 [source code](https://github.com/CWOA/MetricLearningIdentification) [paper](https://arxiv.org/abs/2006.09205).

This project presents a solution to identification of cattle via depth imagery alone, paving the way for robust automated individual classification irrespective of cattle's coat patterns. 

### Installation

Clone this repository to the desired location: `git clone https://github.com/lucy-randewich/cow-depth-identification.git` and
install any missing requirements via `pip` or `conda`: [numpy](https://pypi.org/project/numpy/), [PyTorch](https://pytorch.org/), [OpenCV](https://pypi.org/project/opencv-python/), [Pillow](https://pypi.org/project/Pillow/), [tqdm](https://pypi.org/project/tqdm/), [sklearn](https://pypi.org/project/scikit-learn/), [seaborn](https://pypi.org/project/seaborn/).

### Usage

Dataset loaders are provided for the publicly available OpenCows2020 RGB dataset at: [https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17), and for both of the depth datasets experimented on in the paper, described in chapter 3. These sets are not currently publically available, but could be provided for replication of the paper's results upon request.

An example command for model training is as follows: `python train.py --out_path=output/ --dataset=SixteenBitCows --folds_file=datasets/SixteenBitCows/splits/10-90.json --loss_function=OnlineTripletSoftmaxLoss`, where `python train.py -h` gives explanation of command-line arguments. 


### Citation

Both the OpenSetCows2020 paper and the work on embeddings from Lagunes-Fortiz, M. et al which OpenSetCows2020 itself built upon are given as citations.
```
@article{andrew2020visual,
  title={Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning},
  author={Andrew, William and Gao, Jing and Campbell, Neill and Dowsey, Andrew W and Burghardt, Tilo},
  journal={arXiv preprint arXiv:2006.09205},
  year={2020}
}

@inproceedings{lagunes2019learning,
  title={Learning discriminative embeddings for object recognition on-the-fly},
  author={Lagunes-Fortiz, Miguel and Damen, Dima and Mayol-Cuevas, Walterio},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={2932--2938},
  year={2019},
  organization={IEEE}
}
```