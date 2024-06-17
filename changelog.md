# Changelog

## 2.1.5 (unreleased)

### Bugs Squashed
- Improved `utils_challenge.label_filter` and added few extra tests to ensure that there are never trajectories with segments shorter than `min_seg`.

## 2.1.4

### New Features
- Pathified `utils_challenge.file_nonOverlap_reOrg` (see [#32](https://github.com/AnDiChallenge/andi_datasets/pull/32) and [#37](https://github.com/AnDiChallenge/andi_datasets/pull/37)): now paths are `pathlib.Path` rather than strings.

- Included trap radius as input to `utils_trajectories.plot_trajs` (see [#33](https://github.com/AnDiChallenge/andi_datasets/pull/33)).

- Improved the message when no trajectories are found in the FOV (see [#36](https://github.com/AnDiChallenge/andi_datasets/pull/36)).

### Bugs Squashed

- Improved `utils_challenge.label_filter` and introduced `utils_challenge.unique_labelled` to properly handed filter labelling and avoid segments smaller than `min_seg`. The issue arised because the previous labelling for the `utils_challenge.majority_filter` would vary depending on the actual values in the input array, creating labels based on the sort vector rather than in order of appearance.

- Corrected loop over `self.dics` in `datasets_phenom.create_dataset` (see [#35](https://github.com/AnDiChallenge/andi_datasets/pull/35)).


## 2.1.3
(not released in Pypi)

### New Features
- `ensemble_changepoint_error`: improved the metric such that cases in which groundtruth (GT) has not changepoint (CP) but prediction (pred) has few trajectories with CP does not give maximum error. We consider now that each no GT trajectory correctly predicted is a true positive. For the TP_RMSE, we have changed the case in which, if you had no CP, TP_rmse would be maximal. To be closer to the AnDi2 definition, if there were no TP, we set TP_RMSE = 0. 