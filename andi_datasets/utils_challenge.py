# AUTOGENERATED! DO NOT EDIT! File to edit: source_nbs/utils_challenge.ipynb (unless otherwise specified).

__all__ = ['majority_filter', 'label_filter', 'stepwise_to_list', 'continuous_label_to_list', 'data_to_df',
           'changepoint_assignment', 'changepoint_alpha_beta', 'jaccard_index', 'ensemble_changepoint_error',
           'changepoint_error', 'segment_distance', 'create_binary_segment', 'jaccard_between_segments',
           'segment_assignment', 'metric_anomalous_exponent', 'metric_diffusion_coefficient', 'metric_diffusive_state',
           'check_no_changepoints', 'segment_property_errors', 'extract_ensemble', 'multimode_dist',
           'distribution_distance', 'error_Ensemble_dataset', 'check_prediction_length', 'separate_prediction_values',
           'load_file_to_df', 'error_SingleTraj_dataset']

# Cell
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas
from tqdm.auto import tqdm
import warnings

# Cell
def majority_filter(seq, width):
    offset = width // 2
    seq = [0] * offset + seq
    return [max(set(a), key=a.count)
        for a in (seq[i:i+width] for i in range(len(seq) - offset))]

def label_filter(label, window_size = 5, min_seg = 3):

    if np.min(label) < 0:
        raise ValueError('This function only works with positive labels')

    # if there are no changes:
    if np.sum(label[1:] != label[:-1]) == 0:
        return label

    # define dummy vector with all zeros and ones
    values, dummy = np.unique(label, return_inverse = True)

    # check if there are segment smaller than minimal segment (min_seg)
    cp = np.argwhere(dummy[1:] != dummy[:-1])
    cp = np.append(0, cp)
    current_min = (cp[1:]-cp[:-1]).flatten().min()

    while (current_min < min_seg):

        filt = majority_filter(dummy.tolist(), width = window_size)
        filt = np.array(filt)

        # check if there are segment smaller than minimal segment (min_seg)
        cp = np.argwhere(filt[1:] != filt[:-1])

        # If all changepoints were eliminated
        if cp.size == 0:
            break

        cp = np.append(0, cp)
        current_min = (cp[1:]-cp[:-1]).flatten().min()

        if (dummy == filt).all():
            # If all failed and still have segments smaller than min_seg
            seg_lengths = (cp[1:]-cp[:-1]).flatten().astype(int)
            seg_smaller = np.argwhere(seg_lengths < min_seg).flatten()

            # We over each segment and we asign the values 'by hand'
            for idxsegs in seg_smaller:
                if seg_lengths[idxsegs] == 1:
                    filt[(cp[idxsegs]+1)] = filt[cp[idxsegs]]
                elif seg_lengths[idxsegs] == 2:
                    filt[(cp[idxsegs]+1)] = filt[cp[idxsegs]]
                    filt[(cp[idxsegs]+2)] = filt[cp[idxsegs]+3]

            dummy = filt
            break
        dummy = filt


    # Check boundaries
    if dummy[0] != dummy[1] or dummy[1] != dummy[2]:
        dummy[:2] = dummy[2]
    if dummy[-2] != dummy[-3] or dummy[-1] != dummy[-2]:
        dummy[-3:] = dummy[-3]

    # reset to label values
    dummy_ret = np.zeros_like(dummy).astype(float)

    for idx, v in enumerate(values):
        dummy_ret[dummy == idx] = v

    return dummy_ret

# Cell
def stepwise_to_list(labels):

    l_alpha = labels[:, 0]
    l_D = labels[:, 1]

    # Check if there are changes in any of the variables
    CP_D = np.argwhere(l_D[1:] != l_D[:-1]).flatten()+1
    CP_alpha = np.argwhere(l_alpha[1:] != l_alpha[:-1]).flatten()+1

    # if there are only changes in D
    if CP_D.shape[0] > CP_alpha.shape[0]:
        CP_D = np.append(CP_D, l_D.shape[0])
        Ds = l_d[CP_D.flatten()-1]
        alphas = np.ones_like(Ds)*l_alpha[0]
        return CP_D[:-1], Ds, alphas


    # only changes in alpha
    elif CP_D.shape[0] < CP_alpha.shape[0]:
        CP_alpha = np.append(CP_alpha, l_alpha.shape[0])
        alphas = l_alpha[C_alpha.flatten()-1]
        Ds = np.ones_like(alphas)*l_d[0]
        return CP_alpha[:-1], Ds, alphas

    # Same CP for both
    else:
        CP_D = np.append(CP_D, l_D.shape[0])
        Ds = l_D[CP_D.flatten()-1]
        alphas = l_alpha[CP_D.flatten()-1]
        return CP_D[:-1], Ds, alphas

# Cell
def continuous_label_to_list(labs):
    ''' Given an array of T x 2 labels containing the anomalous exponent and diffusion
    coefficient at each timestep, returns 3 arrays, each containing the changepoints,
    exponents and coefficient, respectively.
    If labs is size T x 3, then we consider that diffusive states are given and also
    return those.
        '''
    # Check if states were given
    are_states = False
    if labs.shape[1] == 3:
        are_states = True

    # Check in which variable there is changes
    CP = np.argwhere((labs[:-1, :] != labs[1:, :]).sum(1) != 0).flatten()+1
    T = labs.shape[0]

    alphas = np.zeros(len(CP)+1)
    Ds = np.zeros(len(CP)+1)
    if are_states: states = np.zeros(len(CP)+1)

    for idx, cp in enumerate(np.append(CP, T)):
        alphas[idx] = labs[cp-1, 0]
        Ds[idx] = labs[cp-1, 1]
        if are_states: states[idx] = labs[cp-1, 2]

    CP = np.append(CP, T)

    if are_states:
        return CP, alphas, Ds, states
    else:
        return CP, alphas, Ds

# Cell
from .utils_trajectories import segs_inside_fov


def data_to_df(trajs,
               labels,
               label_values,
               diff_states,
               min_length = 10,
               fov_origin = [0,0], fov_length= 100, cutoff_length = 10):
    '''
    Inputs:
    :trajs (dimension: T x N x2):
    :labels (dimension: T x N x 2):
    :label_values (array) (size: # of states): values of any property for every existing state.
    :diff_states (array) (size: # of states): labels correspoding to each state as defined in the
    ANDI 2022 state labels: 0: immobile; 1: confined; 2: free diffusion; 3: directed.

    Outputs:
    :df_in (dataframe): dataframe with trajectories
    :df_out (datafram): datafram with label and information
    '''

    xs = []
    ys = []
    idxs = []

    df_out = pandas.DataFrame(columns = ['traj_idx', 'Ds', 'alphas', 'states', 'changepoints'])

    idx_t = 0
    for traj, l_alpha, l_D, l_s in zip(tqdm(trajs), labels[:, :, 0], labels[:, :, 1], labels[:, :, 2]):

        # Check FOV and
        idx_inside_segments = segs_inside_fov(traj, fov_origin, fov_length, cutoff_length)

        if idx_inside_segments is not None:

            for idx_in in idx_inside_segments:
                seg_x = traj[idx_in[0]:idx_in[1], 0]
                seg_y = traj[idx_in[0]:idx_in[1], 1]
                seg_alpha = l_alpha[idx_in[0]:idx_in[1]]
                seg_D = l_D[idx_in[0]:idx_in[1]]
                seg_state = l_s[idx_in[0]:idx_in[1]]

                # Filtering
                seg_alpha = label_filter(seg_alpha)
                seg_D = label_filter(seg_D)
                seg_state = label_filter(seg_state)


                # Stacking data of input dataframe
                xs += seg_x.tolist()
                ys += seg_y.tolist()
                idxs += (np.ones(len(seg_x))*idx_t).tolist()

                # Transforming to list of changepoints and physical properties
                merge = np.hstack((seg_alpha.reshape(seg_alpha.shape[0], 1),
                                   seg_D.reshape(seg_D.shape[0], 1),
                                   seg_state.reshape(seg_state.shape[0], 1)))

                CP, alphas, Ds, states = continuous_label_to_list(merge)

                # Saving each segment info in output dataframe
                df_out.loc[df_out.shape[0]] = [idx_t, Ds, alphas, states, CP]

                # Updating segment index
                idx_t += 1


    # Saving trajectories in Dataframe
    tr_to_df = np.vstack((idxs,
                          xs,
                          ys)).transpose()
    df_in = pandas.DataFrame(tr_to_df, columns = ['traj_idx', 'x', 'y'])

    return df_in, df_out

# Cell
def changepoint_assignment(GT, preds):
    ''' Given a list of groundtruth and predicted changepoints, solves the assignment problem via
    the Munkres algorithm (aka Hungarian algorithm) and returns two arrays containing the index of the
    paired groundtruth and predicted changepoints, respectively.'''

    cost_matrix = np.zeros((len(GT), len(preds)))

    for idxg, gt in enumerate(GT):
        for idxp, pred in enumerate(preds):
            cost_matrix[idxg, idxp] = np.abs(gt-pred)

    return linear_sum_assignment(cost_matrix), cost_matrix

# Cell
def changepoint_alpha_beta(GT, preds, treshold = 10):
    '''Calculate the alpha and beta measure of paired changepoints.
       Inspired from Supplemantary Note 3 in https://www.nature.com/articles/nmeth.2808 '''

    assignment, _ = changepoint_assignment(GT, preds)
    assignment = np.array(assignment)

    threshold = 10
    distance = np.abs(GT[assignment[0]] - preds[assignment[1]])
    distance[distance > threshold] = threshold
    distance = np.sum(distance)

    d_x_phi = threshold*len(GT)
    d_ybar_phi = max([0, (len(preds)-len(GT))*threshold])

    alpha = 1-distance/d_x_phi
    beta = (d_x_phi-distance)/(d_x_phi+d_ybar_phi)

    return alpha, beta

# Cell
def jaccard_index(TP, FP, FN):
    '''Given the true positive, false positive and false negative rates, calculates the Jaccard Index'''
    return TP/(TP+FP+FN)

# Cell
def ensemble_changepoint_error(GT_ensemble, pred_ensemble, threshold = 5):
    ''' Given an ensemble of groundtruth and predicted changepoints, iterates
    over each trajectory's changepoints. For each, it solves the assignment problem
    between changepoints. Then, calculates the RMSE of the true positive pairs and
    the Jaccard index over the ensemble of changepoints (i.e. not the mean of them
    w.r.t. to the trajectories)
    '''

    TP, FP, FN = 0, 0, 0
    TP_rmse = []

    for gt_traj, pred_traj in zip(GT_ensemble, pred_ensemble):

        assignment, _ = changepoint_assignment(gt_traj, pred_traj)
        assignment = np.array(assignment)

        for p in assignment.transpose():

            if np.abs(gt_traj[p[0]] - pred_traj[p[1]]) < threshold:
                TP += 1
                TP_rmse.append((gt_traj[p[0]] - pred_traj[p[1]])**2)
            else:
                FP += 1
                FN += 1

        # Checking false positive and missed events
        if len(pred_traj) > len(gt_traj):
            FP += len(pred_traj) - len(gt_traj)
        elif len(pred_traj) < len(gt_traj):
            FN += len(gt_traj) - len(pred_traj)

    if TP+FP+FN == 0:
        wrn_str = f'No segments found in this dataset.'
        warnings.warn(wrn_str)
        print(threshold)
        return threshold, 0

    # Calculating RMSE
    TP_rmse = np.sqrt(np.mean(TP_rmse))



    return TP_rmse, jaccard_index(TP, FP, FN)

# Cell
def changepoint_error(GT, preds, threshold = 5):
    ''' Given the groundtruth and predicted changepoints for a single trajectory, first solves the assignment problem between changepoints,
    then calculates the RMSE of the true positive pairs and the Jaccard index
    '''

    assignment, _ = changepoint_assignment(GT, preds)
    assignment = np.array(assignment)

    TP, FP, FN = 0, 0, 0
    TP_rmse = []
    for p in assignment.transpose():

        if np.abs(GT[p[0]] - preds[p[1]]) < threshold:
            TP += 1
            TP_rmse.append((GT[p[0]] - preds[p[1]])**2)
        else:
            FP += 1
            FN += 1
    # Calculating RMSE
    TP_rmse = np.sqrt(np.mean(TP_rmse))

    # Checking false positive and missed events
    if len(preds) > len(GT):
        FP += len(preds) - len(GT)
    elif len(preds) < len(GT):
        FN += len(GT) - len(preds)

    return TP_rmse, jaccard_index(TP, FP, FN)

# Cell
def segment_distance(seg1, seg2, epsilon = np.nan):
    dist = np.abs(seg1 - seg2)
    dist[dist > epsilon] = epsilon
    return dist

# Cell
def create_binary_segment(CP, T):
    ''' Given a set of changepoints and the lenght of the trajectory, create segments which are equal to one
    if the segment takes place at that position and zero otherwise '''
    segments = np.zeros((len(CP)+1, T))
    CP = np.append(0, CP)
    for idx, (cp1, cp2) in enumerate(zip(CP[:-1], CP[1:])):
        segments[idx, cp1+1:cp2+1] = 1
    segments[-1, CP[-1]+1:] = 1
    segments[0, 0] = 1
    return segments

# Cell
def jaccard_between_segments(gt, pred):
    '''Given two segments, calculates the Jaccard index between them by considering TP as correct labeling,
    FN as missed events and FP leftover predictions'''

    if len(gt) > len(pred):
        pred = np.append(pred, np.zeros(len(gt) - len(pred)))
    elif len(pred) > len(gt):
        gt = np.append(gt, np.zeros(len(pred) - len(gt)))


    tp = np.sum(np.logical_and(pred == 1, gt == 1))
    fp = np.sum(np.logical_and(pred == 1, gt == 0))
    fn = np.sum(np.logical_and(pred == 0, gt == 1))

    # special case for absence of changepoint
    if tp+fp+fn == 0: return 0
    else: return jaccard_index(tp, fp, fn)

# Cell
def segment_assignment(GT, preds, T = None):
    '''
    Given a list of groundtruth and predicted changepoints, generates a set of segments. Then constructs
    a cost matrix by calculting the Jaccard Index between segments. From this cost matrix, we solve the
    assignment  problem via the Munkres algorithm (aka Hungarian algorithm) and returns two arrays
    containing the index of the groundtruth and predicted segments, respectively.

    If T = None, then we consider that GT and preds may have different lenghts. In that case, the end
    of the segments is the the last CP of each set of CPs.

    '''

    if T is not None:
        T_gt = T_pred = T
        # Check if the GT or predictions are a single integer or an empty array
        if isinstance(GT, int): GT = [GT]
        elif len(GT) == 0: GT = [T-1]

        if isinstance(preds, int): preds = [preds]
        elif len(preds) == 0: preds = [T-1]
    else:
        T_gt = GT[-1]
        if len(GT) > 1:
            GT = GT[:-1]

        T_pred = preds[-1]
        if len(preds) > 1:
            preds = preds[:-1]



    seg_GT = create_binary_segment(GT, T_gt)
    seg_preds = create_binary_segment(preds, T_pred)

    cost_matrix = np.zeros((seg_GT.shape[0], seg_preds.shape[0]))

    for idxg, gt in enumerate(seg_GT):
        for idxp, pred in enumerate(seg_preds):
            cost_matrix[idxg, idxp] = 1-jaccard_between_segments(gt, pred)

    return linear_sum_assignment(cost_matrix), cost_matrix

# Cell
from sklearn.metrics import mean_squared_log_error, f1_score
from .models_phenom import models_phenom

def metric_anomalous_exponent(gt = None, pred = None, max_error = False):
    # Mean absolute error. Maximum error is 2
    if max_error: return 2
    else: return np.mean(np.abs(gt-pred))

def metric_diffusion_coefficient(gt = None, pred = None,
                                 threshold_min = models_phenom().bound_D[0],
                                 max_error = False):
    if max_error:
        return mean_squared_log_error(models_phenom().bound_D[0],
                                      models_phenom().bound_D[1])
    else:
        # considering the presence of zeros and negatives
        pred = np.array(pred).copy(); gt = np.array(gt).copy()
        pred[pred <= threshold_min] = threshold_min
        gt[gt <= threshold_min] = threshold_min
        # mean squared log error
        return mean_squared_log_error(gt, pred)

def metric_diffusive_state(gt = None, pred = None, max_error = False):
    if max_error: return 0
    else: return f1_score(gt.astype(int), pred.astype(int), average = 'micro')

# Cell
def check_no_changepoints(GT_cp, GT_alpha, GT_D, GT_s,
                          preds_cp, preds_alpha, preds_D, preds_s,
                          T = None):
    '''Given predicionts over changepoints and variables, checks if in both GT and preds there is an
    absence of changepoint. If so, takes that into account to pair variables.'''


    if isinstance(GT_cp, int) or isinstance(GT_cp, float):
        GT_cp = [GT_cp]
    if isinstance(preds_cp, int) or isinstance(preds_cp, float):
        preds_cp = [preds_cp]

    no_GT_cp = False; no_preds_cp = False
    # CP always contain the final point of the trajectory, hence minimal length is one
    if len(GT_cp) == 1: no_GT_cp = True
    if len(preds_cp) == 1: no_preds_cp = True


    if no_GT_cp + no_preds_cp == 0:
        return False, None, None, None

    else:

        [row_ind, col_ind], _ = segment_assignment(GT_cp, preds_cp, T)

        if no_GT_cp and not no_preds_cp:
            paired_alpha = np.array([[GT_alpha[0], preds_alpha[col_ind[0]]]])
            paired_D = np.array([[GT_D[0], preds_D[col_ind[0]]]])
            paired_s = np.array([[GT_s[0], preds_s[col_ind[0]]]])

        if no_preds_cp and not no_GT_cp:
            row_position = np.argwhere(col_ind == 0).flatten()[0]
            paired_alpha = np.array([[GT_alpha[row_position], preds_alpha[col_ind[row_position]]]])
            paired_D = np.array([[GT_D[row_position], preds_D[col_ind[row_position]]]])
            paired_s = np.array([[GT_s[row_position], preds_s[col_ind[row_position]]]])

        if no_preds_cp and no_GT_cp:
            paired_alpha = np.array([[GT_alpha[0], preds_alpha[0]]])
            paired_D = np.array([[GT_D[0], preds_D[0]]])
            paired_s = np.array([[GT_s[0], preds_s[0]]])


        return True, paired_alpha, paired_D, paired_s

# Cell
def segment_property_errors(GT_cp, GT_alpha, GT_D, GT_s,
                            preds_cp, preds_alpha, preds_D, preds_s,
                            return_pairs = False,
                            T = None):

    # Check cases in which changepoint where not detected or there were none in groundtruth
    no_change_point_case, paired_alpha, paired_D, paired_s = check_no_changepoints(GT_cp, GT_alpha, GT_D, GT_s,
                                                                                   preds_cp, preds_alpha, preds_D, preds_s, T)

    if not no_change_point_case:
        # Solve the assignment problem
        [row_ind, col_ind], _ = segment_assignment(GT_cp, preds_cp, T)

        # iterate over the groundtruth segments
        paired_alpha, paired_D, paired_s = [], [], []
        for idx_seg, (gt_alpha, gt_D) in enumerate(zip(GT_alpha, GT_D)):

            row_position = np.argwhere(row_ind == idx_seg).flatten()

            # if the GT segment was associated to a prediction
            if len(row_position) > 0:
                row_position = int(row_position)
                # alpha
                gt_a_seg = GT_alpha[idx_seg]
                pred_a_seg = preds_alpha[col_ind[row_position]]
                # d
                gt_d_seg = GT_D[idx_seg]
                pred_d_seg = preds_D[col_ind[row_position]]
                # state
                gt_s_seg = GT_s[idx_seg]
                pred_s_seg = preds_s[col_ind[row_position]]

                paired_alpha.append([gt_a_seg, pred_a_seg])
                paired_D.append([gt_d_seg, pred_d_seg])
                paired_s.append([gt_s_seg, pred_s_seg])

        paired_alpha, paired_D, paired_s = np.array(paired_alpha), np.array(paired_D), np.array(paired_s)

    if return_pairs:
        return paired_alpha, paired_D, paired_s
    else:
        error_alpha = metric_anomalous_exponent(paired_alpha[:,0], paired_alpha[:,1])
        error_D = metric_diffusion_coefficient(paired_D[:,0], paired_D[:,1])
        error_s = metric_diffusive_state(paired_s[:,0], paired_s[:,1])
        return error_alpha, error_D, error_s

# Cell
from .models_phenom import models_phenom
def extract_ensemble(state_label, dic):
        ''' Given an array of the diffusive state and a dictionary with the diffusion information,
        returns a summary of the ensemble properties for the current dataset.

        Args:
            :state_label (array): Array containing the diffusive state of the particles in the dataset.
                                  For multi-state and dimerization, this must be the number associated to the
                                  state (for dimerization, 0 is free, 1 is dimerized). For the rest, we follow
                                  the numeration of models_phenom().lab_state.
            :dic (dictionary):    Dictionary containing the information of the input dataset.
        Returns:
            :ensemble (array):    Matrix containing the ensemble information of the input dataset. It has the
                                  following shape:
                                  |mu_alpha1      mu_alpha2     ... |
                                  |sigma_alpha1   sigma_alpha2  ... |
                                  |mu_D1          mu_D1         ... |
                                  |sigma_D1       sigma_D2      ... |
                                  |counts_state1  counts_state2 ... |
        '''

        # Single state
        if dic['model'] == 'single_state':
            ensemble = np.vstack((dic['alphas'][0],
                                   dic['alphas'][1],
                                   dic['Ds'][0],
                                   dic['Ds'][1],
                                   len(state_label)
                                   ))
        # Multi-state
        if dic['model'] == 'multi_state':
            states, counts = np.unique(state_label, return_counts=True)
            # If the number of visited stated is not equal to the expected number of states
            if len(states) != dic['alphas'].shape[0]:
                states_corrected = np.ones(dic['alphas'].shape[0])
                counts_corrected = np.ones(dic['alphas'].shape[0])
                for s, c in zip(states, counts):
                    counts_corrected[int(s)] = c
            else:
                counts_corrected = counts

            ensemble = np.vstack((dic['alphas'][:, 0],
                                   dic['alphas'][:, 1],
                                   dic['Ds'][:, 0],
                                   dic['Ds'][:, 1],
                                   counts_corrected
                                   ))

        # Immobile
        if dic['model'] == 'immobile_traps':
            counts = [len(state_label[state_label == models_phenom().lab_state.index('i')]),
                      len(state_label[state_label == models_phenom().lab_state.index('f')])]
            ensemble = np.vstack(([0, dic['alphas'][0]],
                                   [0, dic['alphas'][1]],
                                   [0, dic['Ds'][0]],
                                   [0, dic['Ds'][1]],
                                   counts
                                   ))
        # dimerization
        if dic['model'] == 'dimerization':
            counts = [len(state_label[state_label == 0]),
                      len(state_label[state_label == 1])]
            ensemble = np.vstack((dic['alphas'][:, 0],
                                   dic['alphas'][:, 1],
                                   dic['Ds'][:, 0],
                                   dic['Ds'][:, 1],
                                   counts
                                   ))

        if dic['model'] == 'confinement':
            counts = [len(state_label[state_label == models_phenom().lab_state.index('f')]),
                      len(state_label[state_label == models_phenom().lab_state.index('c')])]
            ensemble = np.vstack((dic['alphas'][:, 0],
                                   dic['alphas'][:, 1],
                                   dic['Ds'][:, 0],
                                   dic['Ds'][:, 1],
                                   counts
                                   ))
        return ensemble

# Cell
import scipy.stats
def multimode_dist(params, weights, bound, x):
    func = scipy.stats.truncnorm
    dist = np.zeros_like(x)
    lower, upper = bound

    # If we have single state, change values to list to still
    # have a loop:
    if isinstance(weights, float) or isinstance(weights, int):
        params = [params]
        weights = [weights]

    for param, w in zip(params, weights):
        mean, var  = param
        unimodal = func.pdf(x,
                            (lower-mean)/np.sqrt(var),
                            (upper-mean)/np.sqrt(var),
                            loc = mean,
                            scale = np.sqrt(var))
        dist += w*unimodal
    return dist

# Cell
def distribution_distance(p, q):
#     return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    return np.abs(p-q).mean()

# Cell
from .models_phenom import models_phenom

def error_Ensemble_dataset(true_data, pred_data, return_distributions = False):

    dists = []
    for data in [true_data, pred_data]:

        if len(data.shape) > 1: # If we have more than one state
            alpha_info = np.delete(data, [2,3, -1], 0)
            d_info = data[2:-1,:]
            weights = data[-1,:]
            if weights.sum() > 1: weights /= weights.sum()
        else: # If single state
            alpha_info = data[:2]
            d_info = data[2:-1]
            weights = 1

        for idx, (var, bound) in enumerate(zip([alpha_info, d_info],
                                               [models_phenom().bound_alpha, models_phenom().bound_D])):
            if idx == 0: x = np.linspace(bound[0], bound[1], 1000)
            else: x = np.logspace(np.log10(bound[0]), np.log10(bound[1]), 1000)
            dists.append(multimode_dist(var.T, weights, bound, x))

    # Distance between alpha dists
    distance_alpha = distribution_distance(dists[0], dists[2])
    distance_D = distribution_distance(dists[1], dists[3])

    if return_distributions:
        return distance_alpha, distance_D, dists
    else:
        return distance_alpha, distance_D

# Cell
def check_prediction_length(pred):
    '''Given a trajectory segments prediction, checks whether it has C changepoints and C+1 segments properties values.
    As it must also contain the index of the trajectory, this is summarized by being multiple of 4.
    In some cases, the user needs to also predict the final point of the trajectory. In this case,
    we will have a residu of 1'''
    if len(pred) % 4 == 0 or len(pred) % 4 == 1 :
        return True
    else:
        return False

# Cell
def separate_prediction_values(pred):
    '''Given a trajectory segments prediction, extracts the predictions for each segment property as well as the changepoint values'''
    Ds = pred[1::4]
    alphas = pred[2::4]
    states = pred[3::4]
    cp = pred[4::4]
    return Ds, alphas, states, cp

# Cell
def load_file_to_df(path_file,
                    columns = ['traj_idx', 'Ds', 'alphas', 'states', 'changepoints']):
    '''Given the path of a .txt file, extract the segmentation predictions based on
    the rules of the ANDI Challenge 2022
    '''

    with open(path_file) as f:
        lines_pred = f.read().splitlines()

    df = pandas.DataFrame(columns = columns)

    for line in lines_pred:
        # Extract values with comma separator and transform to float
        pred_traj = line.split(',')
        pred = [float(i) for i in pred_traj]

        # Check that prediction has the correct shape
        pred_correct = check_prediction_length(pred)

        # If correct size, then extract parameters and add it to dataframe
        if pred_correct:
            preds_D, preds_a, preds_s, preds_cp = separate_prediction_values(pred)

            current_row = df.shape[0]
            for param, pred_param in zip(columns, [pred[0], preds_D, preds_a, preds_s, preds_cp]):
                df.loc[current_row, param] = pred_param

    return df

# Cell
def error_SingleTraj_dataset(df_pred, df_true,
                              threshold_error_alpha = 2, max_val_alpha = 2, min_val_alpha = 0,
                              threshold_error_D = 1e5, max_val_D = 1e6, min_val_D = 1e-6, # this is in linear scale
                              threshold_error_s = -1, # this will transform nan into non-existing state
                              threshold_cp = 10,
                              prints = True
                             ):
    '''Given two dataframes, corresponding to the predictions and true labels of a set
    of trajectories from the ANDI challenge 2022, calculates the corresponding metrics
    Columns must be for both (no order needed):
    traj_idx | alphas | Ds | changepoints | states
    df_true must also contain a column 'T'
    '''
    # Initiate counting missing trajectories
    missing_traj = False

    # Deleter saving variables, just in case...
    try: del paired_alpha, paired_D, paired_s
    except: pass

    # for every trajectory, we stack paired segment properties. We also store changepoints info
    ensemble_pred_cp, ensemble_true_cp = [], []
    for t_idx in tqdm(df_true['traj_idx'].values):

        traj_trues = df_true.loc[df_true.traj_idx == t_idx]

        traj_preds = df_pred.loc[df_pred.traj_idx == t_idx]
        if traj_preds.shape[0] == 0:
            # If there is no trajectory, we give maximum error. To do so, we redefine predictions
            # and trues so that they give maximum error
            missing_traj += 1

            preds_cp, preds_alpha, preds_D, preds_s = [[10],
                                                       [0],
                                                       [1],
                                                       [0]]

            trues_cp, trues_alpha, trues_D, trues_s = [[10+threshold_cp],
                                                       [threshold_error_alpha],
                                                       [1+threshold_error_D],
                                                       [10]]

        else:

            preds_cp, preds_alpha, preds_D, preds_s = [np.array(traj_preds.changepoints.values[0]).astype(int),
                                                       traj_preds.alphas.values[0],
                                                       traj_preds.Ds.values[0],
                                                       traj_preds.states.values[0]]

            trues_cp, trues_alpha, trues_D, trues_s = [np.array(traj_trues.changepoints.values[0]).astype(int),
                                                       traj_trues.alphas.values[0],
                                                       traj_trues.Ds.values[0],
                                                       traj_trues.states.values[0]]


        # collecting changepoints
        ensemble_pred_cp.append(preds_cp)
        ensemble_true_cp.append(trues_cp)

        # collecting segment properties error after segment assignment
        pair_a, pair_d, pair_s = segment_property_errors(trues_cp, trues_alpha, trues_D, trues_s,
                                                         preds_cp, preds_alpha, preds_D, preds_s,
                                                         return_pairs = True)



        try:
            paired_alpha = np.vstack((paired_alpha, pair_a))
            paired_D = np.vstack((paired_D, pair_d))
            paired_s = np.vstack((paired_s, pair_s))
        except:
            paired_alpha = pair_a
            paired_D = pair_d
            paired_s = pair_s

    #### Calculate metrics from assembled properties

    # checking for nans and problems in predictions
    wrong_alphas = np.argwhere(np.isnan(paired_alpha[:, 1]) | (paired_alpha[:, 1] > 2) | (paired_alpha[:, 1] < 0)).flatten()
    paired_alpha[wrong_alphas, 1] = paired_alpha[wrong_alphas, 0] + threshold_error_alpha

    wrong_ds = np.argwhere(np.isnan(paired_D[:, 1])).flatten()
    paired_D = np.abs(paired_D)
    paired_D[wrong_ds, 1] = paired_D[wrong_ds, 0] + threshold_error_D

    wrong_s = np.argwhere((paired_s[:, 1] > 4) | (paired_s[:, 1]<0))
    paired_s[wrong_s, 1] = threshold_error_s

    # Changepoints
    rmse_CP, JI = ensemble_changepoint_error(ensemble_true_cp, ensemble_pred_cp, threshold = threshold_cp)

    # Segment properties
    error_alpha = metric_anomalous_exponent(paired_alpha[:,0], paired_alpha[:,1])
    error_D = metric_diffusion_coefficient(paired_D[:,0], paired_D[:,1])
    error_s = metric_diffusive_state(paired_s[:,0], paired_s[:,1])


    if prints:
        print(f'Summary of metrics assesments:')
        if missing_traj is not False:
            print(f'\n{missing_traj} missing trajectory/ies. ')
        print(f'\nChangepoint Metrics \nRMSE: {round(rmse_CP, 3)} \nJaccard Index: {round(JI, 3)}',
              f'\n\nDiffusion property metrics \nMetric anomalous exponent: {error_alpha} \nMetric diffusion coefficient: {error_D} \nMetric diffusive state: {error_s}')



    return rmse_CP, JI, error_alpha, error_D, error_s