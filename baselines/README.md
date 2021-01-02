# JSD Baselines

Run these commands and you will be able to reproduce the paper's numbers.

## Setup

* Install Anaconda (or miniconda).
* Install environment: `conda env create -f environment.yml`
* Activate environment: `conda activate jsd_baselines`

## Equal Distance

* `cd 00_equal-dist`

### Evaluation

* `python run_equal_dist_jsd.py`
* `python run_equal_dist_salami.py`

## Foote

* `cd 01_foote`
* `python run_foote_jsd.py`
* `python run_foote_salami.py`

### Feature Extraction

* Salami:
* JSD:

## CNN

### Feature Extraction

* Salami:
* JSD:

### Training

```
python optimize_peak_picking.py --path_data data/jsd_features/ --path_targets data/jsd_targets/ --path_results results/20200605-084633-cnn_ismir2014_salami-jsd_long/ --path_split ../../splits/jsd_fold-0.yml --bagging 5
```