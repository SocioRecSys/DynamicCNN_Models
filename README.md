# DynamicCNN_Models

## General Overview
This repository contains implementation for the architectures described in the paper: Dynamic CNN Models for Fashion Recommendation in Instagram

### Modules
- `./DataPreprocessing` contains helper functions to read data from Instagram and DeepFashion datasets. The DeepFashion dataset that we used has metadata (categories and attributes), and the Instagram dataset has (categories, sub-categories and attributes). The module also contains methods to partition the datasets into training, validation and testing.  
- `./Datasets` contains multiple folders showing the relations between the datasets metadata. Full details under the module.
- `./Models` contains the implementation of the models described in the paper: DynamicPruning (for single class), DynamicPruning (for multi class), Dynamic Layers (for single class), Dynamic Layers (for multi class) and base archiectures. The evaluation and other helper functions are also under this module. The main files are: singleClass and multiClass. 

The experiments were executed in FloydHub environment 
Example of running single Class expriement with default values in argument parser: floyd run --gpu2 --env pytorch-0.3 --data /useraccount/datasets/instagram-dataset/1:/data 'python singleClass.py'

### Usage 
Implementation was done using PyTorch 0.3 
For detailed information about each module, check readme inside the folders 

For citation: 
### Dynamic CNN Models for Fashion Recommendation in Instagram
[1] Shatha Jaradat, [2] Nima Dokoohaki, [3] Kim Hammar, [4] Ummul Wara, [5] Mihhail Matskin

@inproceedings{jaradat2018dynamic,
  title={Dynamic CNN Models for Fashion Recommendation in Instagram},
  author={Jaradat, Shatha and Dokoohaki, Nima and Hammar, Kim and Wara, Ummul and Matskin, Mihhail},
  booktitle={2018 IEEE Intl Conf on Parallel \& Distributed Processing with Applications, Ubiquitous Computing \& Communications, Big Data \& Cloud Computing, Social Computing \& Networking, Sustainable Computing \& Communications (ISPA/IUCC/BDCloud/SocialCom/SustainCom)},
  pages={1144--1151},
  year={2018},
  organization={IEEE}
}



