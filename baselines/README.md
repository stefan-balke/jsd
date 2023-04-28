# JSD Baselines

Run these commands and you will be able to reproduce the paper's numbers.

## Setup

* Install Anaconda (or miniconda).
* Install environment: `conda env create -f environment.yml`
* Activate environment: `conda activate jsd_baselines`

## Equal Distance

* `cd 00_equal-dist`

### Evaluation

* `python run_equal_dist_salami.py`
    - 0.5 s window: F=0.042, P=0.041, R=0.043
    - 3.0 s window: F=0.237, P=0.231, R=0.244

* `python run_equal_dist_jsd.py`
    - 0.5 s window: F=0.051, P=0.051, R=0.051
    - 3.0 s window: F=0.225, P=0.225, R=0.225

## Foote

* `cd 01_foote`
    
### Feature Extraction

* Salami: `python extract_features_salami.py --src /Volumes/AudioDB/SALAMI/salami-2-0_flat/ --dest data/salami_features --annos ../02_cnn/data/salami_annotations/`
* JSD: `python extract_features_jsd.py --src /Volumes/AudioDB/JSD/wav_orig/ --dest data/jsd_features`

### Evaluation

* `python run_foote_salami.py`
    - Best parameter set for window 0.5
        track_name  kernel_size  threshold  F_mfcc_05  P_mfcc_05  R_mfcc_05  F_mfcc_3  P_mfcc_3  R_mfcc_3 split   wl_ds
    0         inf         40.0        0.0   0.222641   0.226646   0.274298  0.476649  0.466744  0.609715  test  (9, 4)
    - Best parameter set for window 3
        track_name  kernel_size  threshold  F_mfcc_05  P_mfcc_05  R_mfcc_05  F_mfcc_3  P_mfcc_3  R_mfcc_3 split    wl_ds
    21        inf         80.0        0.0   0.168960   0.199428   0.167162  0.462765  0.534235  0.465618  test  (36, 4)

* `python run_foote_jsd.py`
    - Best parameter set for window 0.5
       kernel_size  threshold  F_mfcc_05  P_mfcc_05  R_mfcc_05  F_mfcc_3  P_mfcc_3  R_mfcc_3 split   wl_ds
    4         40.0        0.2      0.192      0.186      0.247     0.454     0.436     0.601  test  (9, 4)
    - Best parameter set for window 3
       kernel_size  threshold  F_mfcc_05  P_mfcc_05  R_mfcc_05  F_mfcc_3  P_mfcc_3  R_mfcc_3 split    wl_ds
    21        80.0        0.0      0.184      0.216      0.185     0.488     0.548     0.505  test  (36, 4)

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
    - 0.5 s window: P=0.389, R=0.389, F=0.365
    - 3.0 s window: P=0.410, R=0.657, F=0.485
* S, short: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220208-231539-cnn_ismir2014_config_salami_short/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.357, R=0.414, F=0.358
    - 3.0 s window: P=0.419, R=0.750, F=0.512 
* S, long: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220209-124533-cnn_ismir2014_config_salami_long/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.234, R=0.223, F=0.213
    - 3.0 s window: P=0.563, R=0.672, F=0.580
* J, short: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220209-175302-cnn_ismir2014_config_jsd_short/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.231, R=0.075, F=0.100
    - 3.0 s window: P=0.432, R=0.420, F=0.386
* J, long: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220210-082246-cnn_ismir2014_config_jsd_long/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.136, R=0.049, F=0.066
    - 3.0 s window: P=0.494, R=0.233, F=0.287
* S+J, short: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220210-130320-cnn_ismir2014_config_salami-jsd_short/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.347, R=0.423, F=0.357
    - 3.0 s window: P=0.484, R=0.660, F=0.522
* S+J, long: `python dnn_testing.py --path_features ../data/cnn_salami_features --path_targets ../data/cnn_salami_targets --path_results results/20220210-215133-cnn_ismir2014_config_salami-jsd_long/ --bagging 5 --test_splits ../data/salami_split.yml --musical_only`
    - 0.5 s window: P=0.242, R=0.226, F=0.221
    - 3.0 s window: P=0.508, R=0.729, F=0.571

#### JSD

* S, short: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220208-231539-cnn_ismir2014_config_salami_short/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.186, R=0.230, F=0.189
    - 3.0 s window: P=0.297, R=0.610, F=0.382
* S, long: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220209-124533-cnn_ismir2014_config_salami_long/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.122, R=0.126, F=0.118
    - 3.0 s window: P=0.423, R=0.579, F=0.465
* J, short: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220209-175302-cnn_ismir2014_config_jsd_short/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.303, R=0.125, F=0.165
    - 3.0 s window: P=0.428, R=0.556, F=0.452
* J, long: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220210-082246-cnn_ismir2014_config_jsd_long/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.193, R=0.117, F=0.139
    - 3.0 s window: P=0.615, R=0.439, F=0.482
* S+J, short: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220210-130320-cnn_ismir2014_config_salami-jsd_short/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.242, R=0.269, F=0.232
    - 3.0 s window: P=0.409, R=0.531, F=0.428
* S+J, long: `python dnn_testing.py --path_features ../data/cnn_jsd_features --path_targets ../data/cnn_jsd_targets --path_results results/20220210-215133-cnn_ismir2014_config_salami-jsd_long/ --bagging 5 --test_splits ../../splits/jsd_fold-0.yml --musical_only`
    - 0.5 s window: P=0.199, R=0.169, F=0.166
    - 3.0 s window: P=0.401, R=0.682, F=0.485
