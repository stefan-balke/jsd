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

* Salami: `python extract_features.py --input --src /Volumes/AudioDB/SALAMI/salami-2-0_flat/ --dest data/cnn_salami_features --annos ../02_cnn/data/salami_annotations/ --targets ../data/cnn_salami_targets`
* JSD: `python extract_features.py --input jsd --src /Volumes/AudioDB/JSD/wav_orig --dest data/cnn_jsd_features --annos ../../data/annotations_csv --targets ../data/cnn_jsd_targets`

### Training

* S, short: `python dnn_training.py --path_data ../data/ --config configs/config_salami_short.yml --bagging 5`
* S, long: `python dnn_training.py --path_data ../data/ --config configs/config_salami_long.yml --bagging 5`
* J, short: `python dnn_training.py --path_data ../data/ --config configs/config_jsd_short.yml --bagging 5`
* J, long: `python dnn_training.py --path_data ../data/ --config configs/config_jsd_long.yml --bagging 5`
* S+J, short: `python dnn_training.py --path_data ../data/ --config configs/config_salami-jsd_short.yml --bagging 5`
* S+J, long: `python dnn_training.py --path_data ../data/ --config configs/config_salami-jsd_long.yml --bagging 5`

### Peak Picker Adjustments

* S, short: `python optimize_peak_picking.py --path_data ../data/ --path_results results/20220208-231539-cnn_ismir2014_config_salami_short/ --bagging 5`
* S, long: `python optimize_peak_picking.py --path_data ../data/ --path_results results/20220209-124533-cnn_ismir2014_config_salami_long/ --bagging 5`
* J, short: `python optimize_peak_picking.py --path_data ../data/ --path_results results/20220209-175302-cnn_ismir2014_config_jsd_short/ --bagging 5`
* J, long: `python optimize_peak_picking.py --path_data ../data/ --path_results results/20220210-082246-cnn_ismir2014_config_jsd_long/ --bagging 5`
* S+J, short: `python optimize_peak_picking.py --path_data ../data/ --path_results results/20220210-130320-cnn_ismir2014_config_salami-jsd_short/ --bagging 5`
* S+J, long: `python optimize_peak_picking.py --path_data ../data/ --path_results results/20220210-215133-cnn_ismir2014_config_salami-jsd_long --bagging 5`

### Testing

#### SALAMI

* S, short, All: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220208-231539-cnn_ismir2014_config_salami_short/ --bagging 5 --test_splits ../data/salami_split.yml`
    - 0.5 s window: P=0.386, R=0.428, F=0.380
    - 3.0 s window: P=0.402, R=0.711, F=0.492
* S, short: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220208-231539-cnn_ismir2014_config_salami_short/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.370, R=0.421, F=0.369
    - 3.0 s window: P=0.441, R=0.749, F=0.528
* S, long: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220209-124533-cnn_ismir2014_config_salami_long/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.245, R=0.230, F=0.222
    - 3.0 s window: P=0.579, R=0.673, F=0.590
* J, short: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220209-175302-cnn_ismir2014_config_jsd_short/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.224, R=0.075, F=0.100
    - 3.0 s window: P=0.452, R=0.429, F=0.402
* J, long: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220210-082246-cnn_ismir2014_config_jsd_long/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.134, R=0.050, F=0.066
    - 3.0 s window: P=0.498, R=0.242, F=0.296
* S+J, short: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220210-130320-cnn_ismir2014_config_salami-jsd_short/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.361, R=0.428, F=0.367
    - 3.0 s window: P=0.503, R=0.661, F=0.536
* S+J, long: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220210-215133-cnn_ismir2014_config_salami-jsd_long/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.244, R=0.225, F=0.223
    - 3.0 s window: P=0.530, R=0.730, F=0.587

#### JSD

* S, short: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220208-231539-cnn_ismir2014_config_salami_short/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.165, R=0.214, F=0.175
    - 3.0 s window: P=0.309, R=0.612, F=0.393
* S, long: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220209-124533-cnn_ismir2014_config_salami_long/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.132, R=0.126, F=0.120
    - 3.0 s window: P=0.423, R=0.566, F=0.462
* J, short: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220209-175302-cnn_ismir2014_config_jsd_short/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.271, R=0.124, F=0.161
    - 3.0 s window: P=0.467, R=0.550, F=0.464
* J, long: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220210-082246-cnn_ismir2014_config_jsd_long/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.181, R=0.111, F=0.132
    - 3.0 s window: P=0.597, R=0.431, F=0.470
* S+J, short: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220210-130320-cnn_ismir2014_config_salami-jsd_short/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.250, R=0.260, F=0.229
    - 3.0 s window: P=0.412, R=0.514, F=0.424
* S+J, long: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220210-215133-cnn_ismir2014_config_salami-jsd_long/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.177, R=0.153, F=0.151
    - 3.0 s window: P=0.405, R=0.669, F=0.486
