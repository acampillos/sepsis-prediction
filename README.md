# Early prediction of sepsis using gradient boosting and deep learning

Sepsis is a life-threatening host response to infection that is associated with high mortality, morbidity, and health costs. Its management is highly time-sensitive because each hour of delayed treatment increases mortality due to irreversible organ damage.

This project aims to analyze sepsis ICU data and predict its onset using machine learning, framing the detection of sepsis as a supervised classification task. Its uses time series data containing laboratory and vital parameters from patients ICU stays.

To determine sepsis labels we use Sepsis-3 definition on an hourly basis, which requires co-occurrence of suspected infection (SI) and organ dysfunction. These events ocurr when:

- Suspicion of infection (SI):
  - If a culture sample was obtained before the antibiotic, then the drug had to be ordered within 72 hours.
  - If the antibiotic was administered first the sampling had to follow within 24 hours.
- Organ dysfunction: When the SOFA score shows an increase of at leats 2 points.

## Data

This project uses [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) database with extraction and preprocessing modified scripts from [Machine Learning and Computational Biology Lab](https://github.com/BorgwardtLab/mgp-tcn). Data files are not provided, as [MIT-LCP](https://lcp.mit.edu/) requires to preserve the patients' privacy. MIMIC-III includes over 58,000 hospital ad-
missions of over 45,000 patients, as encountered between June 2001 and October 2012.

### Extraction and filtering

Patients that fulfill any of these conditions are excluded from the final data set: under the age of 15, no chart data available, logged via CareVue. To ensure that controls cannot be sepsis cases that developed sepsis shortly before ICU, they are required not to be labeled with any sepsis-related ICD-9 billing code.

Cases that develop sepsis earlier than seven hours into their ICU stay are excluded as we aim for an early prediction of the condition. This enables a prediction horizon of 7h.

The final data set contains 570 sepsis cases and 5618 control cases.

### Missing values imputation

Missing values in clinical data is a constant problem that also appears in sepsis prediction. We apply different imputation approaches based on the models trained:

* Gradient boosting models: As time series data cannot be feeded directly, we apply a time series encoding scheme that transform each variable into a set of statistics that represent its distribution. These statistics are: count, mean, std, min, max and quantiles.
* Recurrent neural networks: We impute missing data using forward filling and each variable's median. They also require fixed length data so we apply a padding to each time serie with a masking value that will be ignored during training. 


## Models

We create multiple gradient boosting and recurrent neural networks models (using LSTM and GRU). For each type of model we create base-lines and tuned versions of them. We also create the following more specific models and their tuned versions:

* Gradient boosting with hyperparameters tuned based on the different previous hours of the prediction horizon before the sepsis onset.
* Stacked dense layers and stacked recurrent layers.

We use 44 clinical variables: 15 vital parameters and 29 laboratory parameters. For XGBoost models we obtain 309 variables after encoding (including each time series length).


## Use

1. Install dependencies. Install by running ``pip install -r requirements.txt``. Python version used is 3.8.10.

2. Install PostgreSQL locally. See installation guide for your system in [PostgreSQL downloads](https://www.postgresql.org/download/). PostgreSQL v12.12 and Ubuntu 20.04.4 were used in the project.

3. Accessing and building MIMIC-III. For more details see [MIT Getting Started documentation](https://mimic.mit.edu/docs/gettingstarted/). Steps are:
   1. Become a credentialed user on PhysioNet (https://physionet.org/login/?next=/settings/credentialing/).
   2. Complete required training (https://physionet.org/content/mimiciii/view-required-trainings/1.4/#1).
   3. Sign the required data use agreement (https://physionet.org/login/?next=/sign-dua/mimiciii/1.4/).
   4. Download files from Files section in MIMIC-III website (https://physionet.org/content/mimiciii/1.4/).
   5. Build MIMIC-III locally (https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres).

4. Clone this repository to your local machine. To clone it from the command line run ```git clone https://github.com/acampillos/sepsis-prediction.git```

4. Extract and filter data from MIMIC-III using [Moor et al. scripts](https://github.com/BorgwardtLab/mgp-tcn#query-and-real-world-experiments). 
   1. Download [mgp-tcn repository](https://github.com/BorgwardtLab/mgp-tcn), install its requirements and run ```make query``` inside the project. This step might take several hours.
   2. Once the query has finished, move all files in ```output``` folder to this project's ```output``` folder.

5. Run experiments. Requires creating an folder named ```input``` in the base project folder. Available experiments are in the following table.

Model type                   | Experiment    | Command
---------------------------- | ------------- | -------
Gradient boosting (XGBoost)  | Tuning        | ```python3 ./src/experiments/xgboost_experiments.py tuning```
Gradient boosting (XGBoost)  | Test          | ```python3 ./src/experiments/xgboost_experiments.py test```
Recurrent neural networks    | Tuning        | ```python3 ./src/experiments/rnn_experiments.py tuning```
Recurrent neural networks    | Test          | ```python3 ./src/experiments/rnn_experiments.py test```

## Project structure

	├── configs                               <- Configuration files for the experiments.
	├── input                                 <- Data files used by the models.
	│   ├── rnn
	│   │   ├── test
	│   │   ├── train
	│   │   └── val
	│   │
	│   └── xgboost
	│   	├── test
	│   	├── train
	│   	└── val
	│
	├── logs                                  <- TensorBoard logs.
	│   ├── hyperparam_opt                    <- Hyperparameter tuning logs.
	│   └── train                             <- Training logs.
	│
	├── models                                <- Optimal hyperparameters for the models.
	│
	├── notebooks                             <- Jupyter notebooks with EDA.
	│   ├── files_preview.ipynb
	│   ├── static_variables_eda.ipynb
	│   └── time_series_eda.ipynb
	│
	├── output                                <- Output files from the tuning, training and evaluation.
	│   ├── rnn
	│   └── xgboost
	│
	├── src                                   <- Source code for use in this project.
	│   ├── __init__.py                       <- Makes src a Python module.
	│   │
	│   ├── experiments                       <- Scripts to run the performed experiments.
	│   │   ├── rnn_experiments.py
	│   │   └── xgboost_experiments.py
	│   │
	│   ├── models                            <- Models' implementation.
	│   │
	│   ├── preprocessing                     <- Preprocessing scripts for EDA, training and evaluation.
	│   │   ├── __init__.py                   <- Makes preprocessing a Python module.
	│   │   ├── bin_and_impute.py             <- Binning and imputation of time series' missing values.
	│   │   ├── collect_records.py            <- ICU stays and patients' collection from different data files.
	│   │   ├── main_preprocessing.py         <- Data generation and loading.
	│   │   ├── rnn_preprocessing.py          <- Data padding, loading and label creation.
	│   │   ├── xgboost_preprocessing.py      <- Variables transformation into their stats, loading and label creation.
	│   │   └── util.py
	│   │
	│   ├── visualization                     <- Scripts to create exploratory and results oriented visualizations.
	│   │   ├── __init__.py                   <- Makes visualization a Python module.
	│   │   ├── plots.py                      <- Plots functions used in the project.
	│   │   └── util.py 
	│   │
	│   ├── train_rnn.py
	│   └── train_xgboost.py
	│
	└── requirements.txt                      <- Packages required to reproduce the project's working environment.


## Acknoledgements

* Publication of the final dataset and its related work on sepsis prediction:
   * [Moor, M., Horn, M., Rieck, B., Roqueiro, D., & Borgwardt, K. (2019). Early Recognition of Sepsis with Gaussian Process Temporal Convolutional Networks and Dynamic Time Warping.](http://proceedings.mlr.press/v106/moor19a/moor19a.pdf)
* MIMIC-III and PhysioNet publications and website:
   * [Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. "The MIMIC Code Repository: enabling reproducibility in critical care research." Journal of the American Medical Informatics Association (2017): ocx084.](https://www.ncbi.nlm.nih.gov/pubmed/29036464)
   * Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.
   * [Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet.](https://doi.org/10.13026/C2XW26)
   * Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
* [MIT-LCP code for building MIMIC](https://github.com/MIT-LCP/mimic-code). Available for multiple versions and RDMS.
