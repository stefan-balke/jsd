# DMSA: Data-Driven Music Structure Analysis

This repository contains a reference implementation for a data-driven music structure
analysis approach, in particular, an approach for boundary detection in music recordings.
As for training data, we rely on the SALAMI dataset (version 2), a widely used reference dataset
for this kind of tasks [1].
The used network architecture and pre-processing pipeline are based on the
CNN-based approach presented in [2].

If you use this code, the results, or the pre-trained models in your work, please cite:

> Stefan Balke, Julian Reck, Jan Schlüter, and Meinard Müller:<br />
    DMSA: A CNN Baseline for Data-Driven Boundary Detection in Music<br />
    *Coming soon...*

## Setup

1. Clone repository
1. Install Python dependencies: `conda env create -f environment.yml`

   *Note:* The installation instructions are optimized for Ubuntu 18.04 and Anaconda.
1. Activate Environment: `conda activate dmsa`
1. Download pre-processed features from [Zenodo](https://doi.org/10.5281/zenodo.3256372) and extract ZIP-file to `data/`
1. (Opt.) Run `python -m pytest tests` for unit tests

## Input Features

* Sampling rate 44.1 kHz
* Feature rate: 43 Hz (window length: 2048, hop size: 1024)
* Mel spectrogram:
  - 80 filter
  - [80, 16000] Hz
* Beginnings and ends of songs are padded with pink noise with -70 dB
* (Opt.) Temporal downsampling (3, 6, or 12 frames)

## CNN Architecture (similar to [1])

* Input patch (116 time frames x 80 frequency bins)
* Batch normalization
* Conv Layer: 8 x 6, 16 filters
* Max Pooling Layer: 3 x 6
* Conv Layer: 6 x 3, 32 filters
* Flattening Layer
* Fully connected: 128 outputs
* Fully connected: 1 output ("boundary probability")

## Model Training

For training a CNN ensemble, run the following code:
```
python dnn_training.py --bagging 5 --config configs/config_smearing.yml
```

The CNNs output a novelty function. Extracting boundaries requires additional peak picking step
on this novelty function. The peak picker needs a threshold parameter which we estimate
on the validation dataset.
```
python optimize_peak_picking.py --path_results results/path-to-your-model/ --bagging 5
``` 

## Results

We provide two pre-trained models. Model 1 is working on s of audio
input (feature rate = 7.2 Hz) and Model 2 is working on s of audio input
(feature rate = 3.6 Hz).

|                    | F_0.5 | P_0.5 | R_0.5 | F_3   | P_3   | R_3   |
|--------------------|-------|-------|-------|-------|-------|-------|
| Grill/Schlüter [3] | 0.422 | 0.490 | 0.422 |  ---  |  ---  |  ---  |
| Model 1            | 0.395 | 0.472 | 0.397 | 0.524 | 0.440 | 0.748 |
| Model 2            | 0.172 | 0.156 | 0.212 | 0.586 | 0.531 | 0.731 |


Where F_0.5 relates to as evaluation tolerance of 0.5 seconds and
F_3 to a window of 3 s.

**Model 1:** optimized for 0.5 s, `config_smearing.yml`)<br />
Reproduce results: `python dnn_testing.py --path_results results/20190621-164111-cnn_ismir2014_fine/ --bagging 5`

**Model 2:** optimized for 3.0 s, `config_smearing_coarse.yml`<br />
Reproduce results: `python dnn_testing.py --path_results results/20190624-110556-cnn_ismir2014_coarse/ --bagging 5`


## (Opt.) Feature Extraction

If you have access to the original audio recordings, you can extract
the features yourself by running through the following steps
(otherwise download the extracted features from Zenodo as explained above):

1. Get SALAMI annotations: `git submodule init`
1. Prepare annotations: `python salami_prepare_annotations.py`
1. Extract features from audio and targets from annotations:
    ```
    python extract_features.py\
        -s path/to/salami_audios/\
        -a data/salami_annotations
    ```

## Acknowledgments

t.b.d.

## Literature

> [1] Jordan B. L. Smith, J. Ashley Burgoyne, Ichiro Fujinaga, David De Roure, and J. Stephen Downie:<br />
    Design and creation of a large-scale database of structural annotations.<br />
    Proceedings of the International Society for Music Information Retrieval Conference (ISMIR),<br />
    Miami, FL. 2011<br />
    URL: https://doi.org/10.5281/zenodo.1416883

> [2] Karen Ullrich, Jan Schlüter, and Thomas Grill:<br />
    Boundary Detection in Music Structure Analysis using Convolutional Neural Networks.<br />
    In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR),<br />
    Taipei, Taiwan, 2014.<br />
    URL: https://doi.org/10.5281/zenodo.1415885.

> [3] Thomas Grill and Jan Schlüter:<br />
    Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations.<br />
    In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR),<br />
    Málaga, Spain, 2015.<br />
    URL: https://doi.org/10.5281/zenodo.1417460