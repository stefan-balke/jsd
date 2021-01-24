# Jazz Structure Dataset

* Authors: Stefan Balke, Julian Reck, and Meinard Müller
* Special thanks to Moritz Berendes as the main annotator!

# Abstract

The Jazz Structure Dataset (JSD) comprises structure annotations for 343
famous Jazz recordings. Along with the temporal annotations for song
regions (e.g., Solo 1 starting at 40 s and ending at 113 s), it provides
further metadata about the predominant instrument (in most cases the soloist)
and the accompanying instruments (e.g., drums and piano).

# Technical Description

## Annotation Process

* TODO: Add short explanation of annotation process

## Installation

1. `conda env create -f environment.yml`
2. `conda activate jsd`

## Folder Structure

* `data/annotations_sv`: Original annotation files. Can be opened with SonicVisualiser.
* `data/annotations_raw`: Direct export from SonicVisualiser into a textual format.
* `data/annotations_csv`: Final annotations in CSV format.
* `data/annotations_jams`: Final annotations in Jams format.

## Usage

If you want to build the annotations from ground up or integrate changes,
please call `python prepare_annotations.py`

This script takes the files from `data/annotations_raw` and applies the following modifications:

 * Add silence regions at the beginning and end (based on the track durations obtained from `data/track_durations.csv`).
 * Unify notation into a region-like notation with start and end time points.
 * Fix small holes in the annotations (e.g., annotation starts 0.001 s after previous annotation ends).

## Tools

* `python general_statistics`
* Output can be found in `data/general_statistics`.

## Remarks

* The SonicVisualiser files of the following files are missing, however, annotations are in `annotations_raw`:
  - CharlieParker_K.C.Blues_Orig
  - SidneyBechet_I'mComingVirginia

* The following tracks are contained in the WJD but are actually duplicates
  - PatMetheny_CabinFever_Orig == MichaelBrecker_CabinFever_Orig
  - PatMetheny_MidnightVoyage_Orig == MichaelBrecker_MidnightVoyage_Orig
  - PatMetheny_NothingPersonal_Orig == MichaelBrecker_NothingPersonal_Orig
  --> we only keep the original files (in these cases Michael Brecker)
