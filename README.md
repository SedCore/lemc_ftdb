# LEMC-FTDB
Code of the Learned-Weight Monte Carlo FT-DropBlock (LEMC-FTDB) with per-model temperature scaling. The original paper is published in open-access here: [https://ieeexplore.ieee.org/document/11271584](https://ieeexplore.ieee.org/document/11271584)

## Abstract
Reliable predictive uncertainty—in particular predictive entropy used at decision time—in motor imagery electroencephalography (EEG)-based brain-computer interfaces (BCIs) is critical for safe real-world operation. Although Monte Carlo Dropout and deep ensembles are effective for modeling predictive uncertainty, structured regularization approaches—such as FT-DropBlock for EEG-targeted convolutional neural networks (CNNs)—remain insufficiently investigated. We present Learned-Weight Ensemble Monte Carlo FT-DropBlock (LEMC-FTDB), a model-agnostic generalizable framework that integrates structured regularization, stochastic inference, learned aggregation, and per-model temperature scaling to deliver calibrated probabilities and informative uncertainty estimates on CNN models. We evaluated our framework on the EEGNet and EEG-ITNet backbones using the BCI Competition IV 2a dataset, with metrics including predictive entropy, mutual information, expected calibration error, negative log-likelihood, Brier score, and misclassification-detection AUC. Results show that LEMC-FTDB consistently improves probability quality and calibration, strengthens misclassification detection, and maintains superior correctness and Cohen’s kappa versus Monte Carlo-based, deep ensemble-based, and deterministic baselines. Crucially, predictive entropy cleanly ranks trial reliability and enables an hold/reject policy that achieves lower risk at the same coverage in risk–coverage analyses, supporting practical deployment. Our code is available at: github.com/SedCore/lemc_ftdb.

## Environment
We ran our experiment in the following environment:
* NVIDIA Tesla T4 GPU with 16GB RAM
* Linux Ubuntu 24.04 LTS operating system
* CUDA 12.9 library
* Python 3.12.3 x64
* Tensorflow 2.19.0 with XLA compiler

## Requirements
* braindecode==0.8.1
* moabb==1.2.0
* matplotlib==3.10.3
* numpy==1.26.4
* scikit_learn==1.6.1
* skorch==1.0.0
* tensorflow==2.19.0
* torch==2.8.0+cu129
* keras==3.10.0

## Usage
### Models tested
* EEGNet [(GitHub repo)](https://github.com/vlawhern/arl-eegmodels). Original paper [here](https://doi.org/10.1088/1741-2552/aace8c).
* EEG-ITNet [(GitHub repo)](https://github.com/AbbasSalami/EEG-ITNet). Original paper [here](https://doi.org/10.1088/1741-2552/aace8c).

### Dataset
We used the [BCI Competition IV 2a](https://www.bbci.de/competition/iv) dataset, imported in our code using the [MOABB](https://doi.org/10.5281/zenodo.10034223) library.

### Running the models
Run the main.py file. The use of parameters is optional since there are default values, unless you need to choose other values.
```
python3 main.py --model=EEGNetv4
```
Parameters:\
--model: CNN model. Choices: EEGNetv4 (default), EEGITNet.\
--prob: Overall drop probability: 0.2, 0.3, 0.4, 0.5 (default), 0.6, 0.7, 0.8, 0.9\
--block: Block size value for FT-DropBlock: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 (default), etc.
--subject: Select the subject ID (1-9) or 10 (default) to load all from the dataset.

## Paper citation
If you use our code and found it helpful, please cite our paper:
```
@ARTICLE{lemc_ftdb,
  author={Nzakuna, Pierre Sedi and Gallo, Vincenzo and CarratÙ, Marco and Paciello, Vincenzo and Pietrosanto, Antonio and Lay-Ekuakille, Aimé},
  journal={IEEE Open Journal of Instrumentation and Measurement}, 
  title={Learned-Weight Ensemble Monte Carlo DropBlock for Uncertainty Estimation and EEG Classification}, 
  year={2025},
  doi={10.1109/OJIM.2025.3638922}}
```
## License
Please refer to the LICENSE file.
