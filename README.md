
# Diagnosis of Glioblastoma Multiforme Progression via Interpretable Structure-Constrained Graph Neural Networks

This repository holds the PyTorch code of our IEEE TMI paper *Diagnosis of Glioblastoma Multiforme Progression via Interpretable Structure-Constrained Graph Neural Networks*. 

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (**Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University**) preserve the copyright and all legal rights of these codes.


## Author List

Xiaofan Song, Jun Li, Xiaohua Qian


## Abstract

Glioblastoma multiforme (GBM) is the most common type of brain tumors with high recurrence and mortality rates. After chemotherapy treatment, GBM patients still show a high rate of differentiating pseudoprogression (PsP), which is often confused as true tumor progression (TTP) due to high phenotypical similarities. Thus, it is crucial to construct an automated diagnosis model for differentiating between these two types of glioma progression. However, attaining this goal is impeded by the limited data availability and the high demand for interpretability in clinical settings. In this work, we propose an interpretable structure-constrained graph neural network (ISGNN) with enhanced features to automatically discriminate between PsP and TTP. This network employs a metric-based meta-learning strategy to aggregate class-specific graph nodes, focus on meta-tasks associated with various small graphs, thus improving the classification performance on small-scale datasets. Specifically, a node feature enhancement module is proposed to account for the relative importance of node features and enhance their distinguishability through inductive learning. A graph generation constraint module enables learning reasonable graph structures to improve the efficiency of information diffusion while avoiding propagation errors. Furthermore, model interpretability can be naturally enhanced based on the learned node features and graph structures that are closely related to the classification results. Comprehensive experimental evaluation of our method demonstrated excellent interpretable results in the diagnosis of glioma progression. In general, our work provides a novel systematic GNN approach for dealing with data scarcity and enhancing decision interpretability. Our source codes will be released at https://github.com/SJTUBME-QianLab/GBM-GNN.


## Requirements

* `pytorch 1.1.0`
* `numpy 1.17.2`
* `python 3.6.1`


## How to train

Run ```pre_train.py``` first to pre-train CNN for extracting features.

Then run ```train_gnn_cnn.py``` to get final model.



## Citing the Work

If you find our code useful in your research, please consider citing:

```
@ARTICLE{9868065,
  author={Song, Xiaofan and Li, Jun and Qian, Xiaohua},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Diagnosis of Glioblastoma Multiforme Progression via Interpretable Structure-Constrained Graph Neural Networks}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3202037}}
```

## Contact

For any question, feel free to contact

```
Jun Li : dirk_li@sjtu.edu.cn
```


## Acknowledgements

This code is developed on the code base of [few-shot-gnn](https://github.com/vgsatorras/few-shot-gnn) and [population-gcn](https://github.com/parisots/population-gcn). Many thanks to the authors of these works.  