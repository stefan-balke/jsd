# CNN-based Boundary Detection

This repository contains a reference implementation for a data-driven music structure
analysis approach, in particular, an approach for boundary detection in music recordings.
As for training data, we rely on the SALAMI dataset (version 2), a widely used reference dataset
for this kind of tasks [1].
The used network architecture and pre-processing pipeline are based on the
CNN-based approach presented in [2].

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