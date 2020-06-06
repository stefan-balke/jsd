# Baselines

Run these commands and you will be able to reproduce the paper's numbers.

## Equal Distance

## Foote

## CNN

### Training

```
python optimize_peak_picking.py --path_data data/jsd_features/ --path_targets data/jsd_targets/ --path_results results/20200605-084633-cnn_ismir2014_salami-jsd_long/ --path_split ../../splits/jsd_fold-0.yml --bagging 5
```