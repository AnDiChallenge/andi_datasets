# Changelog

## Unreleased

## Added
- 2.1.3: changed CP metric such that cases in which GT has not CP but pred has few trajectories with CP does not give maximum error. We consider now that each no GT trajectory correctly predicted is a true positive. For the TP_RMSE, we have changed the case in which, if you had no CP, TP_rmse would be maximal. To be closer to our definition, see that in the previous case there was no TP, hence to be fairer we set TP_RMSE = 0. 