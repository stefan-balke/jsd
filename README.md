# Jazz Structure Dataset (JSD)

This repository contains the corresponding code for the article:
>[Stefan Balke](https://stefan.balke.at),
>Julian Reck,
>[Christof Weiß](https://www.audiolabs-erlangen.de/fau/assistant/weiss),
>[Jakob Abeßer](https://www.idmt.fraunhofer.de/en/institute/doctorands/abesser.html),
> and [Meinard Müller](https://www.audiolabs-erlangen.de/fau/professor/mueller):<br>
[JSD: A Dataset for Structure Analysis in Jazz Music](#),<br>
*Transactions of the International Society for Music Information Retrieval*, 2022.

# Abstract

The Jazz Structure Dataset (JSD) comprises structure annotations for 340
famous Jazz recordings. Along with the temporal annotations for song
regions (e.g., Solo 1 starting at 40 s and ending at 113 s), it provides
further metadata about the predominant instrument (in most cases the soloist)
and the accompanying instruments (e.g., drums and piano).

This repository contains the annotations and reference implementations for the baselines described in the paper.

# Technical Description

## Installation

1. `conda env create -f environment.yml`
2. `conda activate jsd`

## Folder Structure

* `data/annotations_sv`: Original annotation files. Can be opened with SonicVisualiser.
* `data/annotations_raw`: Direct export from SonicVisualiser into a textual format.
* `data/annotations_csv`: Final annotations in CSV format.

## Usage

If you want to build the annotations from ground up or integrate changes,
please call `python prepare_annotations.py`

This script takes the files from `data/annotations_raw` and applies the following modifications:

 * Add silence regions at the beginning and end (based on the track durations obtained from `data/track_durations.csv`).
 * Unify notation into a region-like notation with start and end time points.
 * Fix small holes in the annotations (e.g., annotation starts 0.001 s after previous annotation ends).

The usage of the baselines is explained in a separate Readme in the `baselines` folder.

## Tools

* `python general_statistics`
* Output can be found in `data/general_statistics`.

## Remarks

* The following tracks are contained in the WJD but are actually duplicates
  - PatMetheny_CabinFever_Orig == MichaelBrecker_CabinFever_Orig
  - PatMetheny_MidnightVoyage_Orig == MichaelBrecker_MidnightVoyage_Orig
  - PatMetheny_NothingPersonal_Orig == MichaelBrecker_NothingPersonal_Orig
  --> we only keep the original files (in these cases Michael Brecker)
* CannonballAdderley_ThisHere_Orig has applause and an intro speech at the beginning an applause at the end which is not annotated as a separate section
* If you use the dataset in your work, please cite the above mentioned paper.
* In case you have questions, please feel free to reach out in an issue or by mail!
