# Deep Vecchia Ensemble
This repository contains code for our paper [Vecchia Gaussian Process Ensembles on Internal Representations of Deep Neural Networks](https://arxiv.org/abs/2305.17063) (AISTATS 2025) by Felix Jimenez and Matthias Katzfuss.
 
## Overview
The Deep Vecchia Ensemble (DVE) is a scalable method for deterministic uncertainty quantification (UQ) in deep neural networks (DNNs). DVE builds an ensemble of Gaussian processes (GPs) on the hidden-layer outputs of a pretrained DNN. By using Vecchia approximations to exploit nearest-neighbor conditional independence, DVE efficiently handles large datasets while avoiding the need to retrain the network. This approach mitigates feature collapse issues common in deterministic UQ methods and provides interpretable uncertainty estimates by identifying training points similar to the test point. DVE achieves strong performance across benchmark datasets and complex prediction tasks. 


<img src="/figs/dve_intro.jpg" alt="DVE Idea" width="350">

 **Different layers imply different nearest neighbors.** Left: The input (red star) has different nearest-neighbor conditioning sets based on the metrics induced by the layers of the DNN, with the magenta point being in two of the three conditioning sets. The brown, blue, and pink shaded areas denote the regions in input space that will be mapped to a hypersphere in the first, second, and third intermediate spaces, respectively. The conditioning sets derived from the different regions may overlap, as in the blue and brown region, or be disjoint from the others as in the pink region. Right: The labeled training data are propagated through the network and intermediate feature maps are stored. For a red test point, we assess uncertainty by considering and weighting instances in the training data that are similar to the test sequence in one or more of the feature maps.


## Workflow Summary
<img src="/figs/dve_model_summary.jpg" alt="DVE Idea" width="350">

**The DVE pipeline follows a straightforward design.** The network maps inputs to intermediate spaces, where nearest neighbors are identified. At each layer, the responses from these neighboring points are combined to form a distribution over the response. The distributions from all layers are then merged to produce a unified predictive distribution.

## Repository Structure
- **`/experiments`** - An example of training and evaluating the DVE on UCI regression tasks.
- **`/code`** - Our code base.
- **`/container`** - Material for building our  Docker image. 

## Citation
If you find this work useful, please cite:

```
@inproceedings{
    jimenez2025dve, 
    title={Vecchia Gaussian Process Ensembles on Internal Representations of Deep Neural Networks}, 
    author={Felix Jimenez and Matthias Katzfuss}, 
    booktitle={Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS)}, 
    year={2025}
}
```