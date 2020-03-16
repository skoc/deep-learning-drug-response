# deep-learning-drug-response

![](https://img.shields.io/badge/Status-under--dev-red.svg) ![](https://img.shields.io/badge/Python-3.7-green.svg)![License: MIT](https://img.shields.io/github/license/skoc/deep-learning-drug-response.svg)

Deep Learning solutions for drug response prediction in public omics datasets.

## Description
Briefly, first feature selection with Autoencoder (for extracting feature subspace) or Lasso (for variable selection) then fully connected network is used as a predictive model to the extracted feature subspace by step one for final prediction of selected chemotherapy drugs.

Details will be updated...

### Dataset

Dataset is publicly available. Currently Gene Expression data is used for the drug response prediction. In the near future, planning to add two more omics files (CNV, Mutation) to improve the prediction achieved with Gene Expression data.

* [Genomics of Drug Sensitivity in Cancer (GDSC)](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3610/files/raw/)
    * Samples supports multi-omics profiles.
* ...


### Run on Google Colab
Run the pipeline on Google Colab easily:
[Deep Learning Drug Response Colab](https://github.com/skoc/deep-learning-drug-response/blob/master/notebooks/dl_drug_response_colab.ipynb)

### Pipeline hyperparameter configuration

Tweak training parameters with file placed in `config/setup.yml`