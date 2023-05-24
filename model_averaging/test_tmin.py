#!/usr/bin/env python3

from .stats import *
from .synth_data import *
from .fitting import *
from .plotting import *
# import matplotlib.pyplot as plt

def cut_data_tmin(data, tmin, tmax=None, data_snr_min=0.0, continuous_cut=False):
    """
    Cuts data to only include data above tmin
    (and below tmax, if provided.)
    Args:
      data: Dictionary of data
      tmin: Minimum t-value to cut at (not inclusive, i.e.
            t > tmin is kept.)
      tmax: (optional) Maximum t-value to cut at (also not
            inclusive.)
      data_snr_min: if greater than zero, all data with signal-to-noise
            ratio less than the cut are eliminated.
      continuous_cut: if True, all data above the lowest-t point where 
            SNR < data_snr_min are removed.  if False, individual data
            points are removed based on SNR (so the final data set may
            not be continuous in t.)

    Returns:
      An abbreviated synth_data dictionary with the cuts applied.
    """
    
    T_keep = data['t'] >= tmin
    T_cut = np.logical_not(T_keep)
    T_cut[0] = False
    
    if tmax is not None:
        T_keep = np.logical_and(T_keep, data['t'] <= tmax)
        T_cut = np.logical_and(T_cut, data['t'] <= tmax)

    if data_snr_min > 0.0:
        data_snr = np.abs(gv.mean(data['y']) / gv.sdev(data['y']))

        if continuous_cut:
            noise_cut_index = np.argmax(data_snr < data_snr_min)
            data_noise_cut = data['t'] < data['t'][noise_cut_index]
        else:
            data_noise_cut = data_snr > data_snr_min

        T_keep = np.logical_and(T_keep, data_noise_cut)
        T_cut = np.logical_and(T_cut, data_noise_cut)
    
    yexact = data.get('yexact', None)
    if yexact is not None:
        yexact_keep = yexact[T_keep]
        yexact_cut = yexact[T_cut]
    else:
        yexact_keep = None
        yexact_cut = None

    return {
        'ND': data['ND'],
        't': data['t'][T_keep],
        'y': data['y'][T_keep],
        'yexact': yexact_keep,
        'yraw': data['yraw'][:,T_keep],
        't_cut': data['t'][T_cut],
        'y_cut': data['y'][T_cut],
        'yexact_cut': yexact_cut,
        'yraw_cut': data['yraw'][:,T_cut],
    }


def test_poly_tmin_SE(
    test_data,
    k=1,
    Nt=32,
    max_tmin=28,
    min_t_range=4,
    tmax=None,
    obs_name="a0",
    data_snr_min=0.0,
    priors_SE=None,
    double_exp_fit=False,
    continuous_cut=False,
    IC_list=None,
):
    """
    Test a fixed single-exponential model against the given data with a sliding t_min cut, extracting the given
    common fit parameter using model averaging.
    Args:
      test_data: Data set to run fits against.
      k: order of polynomial fit
      Nt: Parameter for exponential model/data
      max_tmin: Maximum value of tmin to use (minimum is 0.)
      min_t_range: Minimum number of t values to be fitted.
      tmax: maximum t value to be fitted.
      obs_name: Common fit parameter to study with model averaging (default: 'E0', the ground state energy.)
      data_snr_min: minimum allowed signal-to-noise ratio for the data.
      priors_SE: prior dictionary to use for the fits - use defaults if none provided.
      double_exp_fit: if True, runs two-state instead of one-state fits.
      continuous_cut: whether the data_snr_min should be continuous or not, see documentation for cut_data_tmin above.
      IC_list: Compute the listed ICs if provided (otherwise, compute a default set.)
    """

    # Need at least 1 dof to fit with
    assert max_tmin < Nt - 2

    T_test = np.arange(1, Nt + 1)

    # Run fits of synthetic data vs. tmin
    obs_vs_tmin = []
    fit_vs_tmin = []
    Q_vs_tmin = []

    if IC_list is None:
        IC_list = ['BAIC_perf', 'BAIC_sub']


    prob_vs_tmin = {}
    IC_vs_tmin = {}
    for IC in IC_list:
        prob_vs_tmin.update({IC: []})
        IC_vs_tmin.update({IC: []})

    
    tmin_array = np.array([])

    fit_func = run_fit_poly

    for tmin in T_test[:max_tmin]:

        if tmax is None:
            cut_test_data = cut_data_tmin(test_data, tmin, data_snr_min=data_snr_min, continuous_cut=continuous_cut)
        else:
            cut_test_data = cut_data_tmin(test_data, tmin, tmax=tmax, data_snr_min=data_snr_min, continuous_cut=continuous_cut)
        if tmin == T_test[0]:
            returned_data = cut_test_data
        
        if tmin == np.amin(T_test[:max_tmin]):
            this_fit, this_design_mat = run_fit_poly(cut_test_data, Nt, k)
        else:
            this_fit, this_design_mat = run_fit_poly(cut_test_data, Nt, k)

        
        # Safety: do not attempt fits with no data
        if len(cut_test_data['y']) == 0:
            continue

        if len(cut_test_data['t']) < min_t_range:
            continue

        tmin_array = np.append(tmin_array,tmin)
        fit_vs_tmin.append(this_fit)
        obs_vs_tmin.append(this_fit.p[obs_name])
        Q_vs_tmin.append(this_fit.Q)
        
        ICs = {}
        for IC in IC_list:
            ICs.update({IC: None})
            ICs[IC] = get_model_IC(
                this_fit,
                cut_test_data,
                design_mat=this_design_mat,
                return_prob=False,
                IC=IC,
            )
            
            IC_vs_tmin[IC].append(ICs[IC])
       
        
    for IC in IC_list:
        try:
            prob_vs_tmin[IC] = np.exp(-(IC_vs_tmin[IC] - np.amin(IC_vs_tmin[IC]))/2)
        except ValueError:
            pass
        except TypeError:
            pass
        
    # Compute model-averaged result
    obs_avg_IC = {}
    for IC in IC_list:
        obs_avg_IC.update({IC: model_avg(obs_vs_tmin, prob_vs_tmin[IC])})
    
    # Return results as a dictionary
    return {
        "tmin": tmin_array,
        "data": returned_data,
        "fits": fit_vs_tmin,
        "obs": obs_vs_tmin,
        "prob": prob_vs_tmin,
        "IC": IC_vs_tmin,
        "obs_avg_IC": obs_avg_IC,
        "Qs": Q_vs_tmin,
    }