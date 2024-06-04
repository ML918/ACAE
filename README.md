# ACAE
ACAE: An adversarial contrastive autoencoder for robust multivariate time series anomaly detection

The main contributions of ACAE are as follows:

1.An adversarial contrastive autoencoder framework for robust multivariate time series anomaly detection.

2.A proxy task based on the feature combination and decomposition for MTS contrastive learning.

3.A multi-scale timestamp mask-based MTS data augmentation method.

![image](https://github.com/yujiahao12138/ACAE/assets/62507173/b56cb215-4315-434b-9661-c9c6a95efb7d)


## Get Started
1.Requirements: Python 3.8, PyTorch 1.12. 

2.Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.

3.Train and evaluate. You can reproduce the experiment results as follows:
```bash
python main.py --data 'SWAT'
python main.py --data 'SMD'
python main.py --data 'PSM'
python main.py --data 'MSL'
python main.py --data 'SMAP'
```


## Main Result
We compare our model with 14 baselines. **Generally,  ACAE achieves SOTA.**

![image](https://github.com/yujiahao12138/ACAE/assets/62507173/c07e5a28-35ac-4ba4-ba65-b02d7a7096db)


## Citation
If you find this repo useful, please cite our paper. 

```
@article{YU2024123010,
title = {An adversarial contrastive autoencoder for robust multivariate time series anomaly detection},
journal = {Expert Systems with Applications},
volume = {245},
pages = {123010},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.123010},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423035121},
author = {Jiahao Yu and Xin Gao and Feng Zhai and Baofeng Li and Bing Xue and Shiyuan Fu and Lingli Chen and Zhihang Meng}
}
```
