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

* Salami: `python extract_features_salami.py --src /Volumes/AudioDB/SALAMI/salami-2-0_flat/ --dest data/salami_features --annos ../02_cnn/data/salami_annotations/`
* JSD: `python extract_features_jsd.py --src /Volumes/AudioDB/JSD/wav_orig/ --dest data/jsd_features`

### Evaluation

* Salami:
* JSD:

## CNN

### Feature Extraction

* Salami:
* JSD:

### Training

* S, short: `python dnn_training.py --path_data data/sbalke/jsd/jsd_data/ --config configs/config_salami_short.yml --bagging 5`
* S, long: `python dnn_training.py --path_data data/sbalke/jsd/jsd_data/ --config configs/config_salami_long.yml --bagging 5`
* J, short: `python dnn_training.py --path_data data/sbalke/jsd/jsd_data/ --config configs/config_jsd_short.yml --bagging 5`
* J, long: `python dnn_training.py --path_data data/sbalke/jsd/jsd_data/ --config configs/config_jsd_long.yml --bagging 5`

### Peak Picker Adjustments

* S, short: `python optimize_peak_picking.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20201230-212634-cnn_ismir2014/ --bagging 5`
* S, long: `python optimize_peak_picking.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20210101-142319-cnn_ismir2014/ --bagging 5`
* J, short: `python optimize_peak_picking.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20210101-195946-cnn_ismir2014 --bagging 5`
* J, long: TODO `python optimize_peak_picking.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20210102-144947-cnn_ismir2014 --bagging 5`
* S+J, short:
* S+J, long:

### Testing

* S, short: `python dnn_testing.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20201230-212634-cnn_ismir2014/ --bagging 5`
* S, long: `python dnn_testing.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20210101-142319-cnn_ismir2014/ --bagging 5`
* J, short: `python dnn_testing.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20210101-195946-cnn_ismir2014 --bagging 5`
* J, long: TODO `python dnn_testing.py --path_data data/sbalke/jsd/jsd_data/ --path_results results/20210102-144947-cnn_ismir2014 --bagging 5`
* S+J, short:
* S+J, long:
