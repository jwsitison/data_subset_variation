#!/usr/bin/env python3

import numpy as np
import gvar as gv
import jax
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


# Models, in the style of the 'lsqfit' Python module
def poly_model_lsqfit(t, p, Nt, m=2):
    ans = 0.0
    for i in range(0, m + 1):
        if i==0 or i==1:
            ans += p[f"a{i}"] * (-(t-Nt) / Nt) ** i
        else:
            ans += p[f"a{i}"] * (-(t-Nt) / Nt) ** i
    return ans

def single_exp_model_lsqfit(t, p):
    return p["A0"] * np.exp(-p["E0"] * t)

def double_exp_model_lsqfit(t, p):
    ans = 0.0
    ans += p["A0"] * np.exp(-p["E0"] * t)
    ans += p["A1"] * np.exp(-(p["E0"] + np.exp(p["ldE1"])) * t)
    return ans

def multi_exp_model_lsqfit(t, p, Nexc=2):
    ans = 0.0
    for i in range(0, Nexc):
        if i==0:
            ans += p["A{}".format(i)] * np.exp(-p["E{}".format(i)] * t)
        else:
            ans += p["A{}".format(i)] * np.exp(-p["E{}".format(i)] * t)
    return ans


# Models, in the style of the 'jax' Python module
def poly_model_jax(p, t, Nt, m=2):
    design_mat = jnp.power(-(t-Nt) / Nt, jnp.array([jnp.arange(0, m + 1)]).T).T
    return design_mat @ p

@jax.jit
def single_exp_model_jax(p, t):
    # p must be entered as p = [A0, E0]
    return p[0] * jnp.exp(-p[1] * t)

d1_single_exp = jax.jit(jax.jacfwd(single_exp_model_jax))
d2_single_exp = jax.jit(jax.jacfwd(d1_single_exp))
d3_single_exp = jax.jit(jax.jacfwd(d2_single_exp))

single_exp_model_derivs = {
    'd1': d1_single_exp,
    'd2': d2_single_exp,
    'd3': d3_single_exp,
}

@jax.jit
def double_exp_model_jax(p, t):
    # p must be entered as [A0, E0, A1, ldE1]
    ans = 0.0
    ans += p[0] * jnp.exp(-p[1] * t)
    ans += p[2] * jnp.exp(-(p[1] + jnp.exp(p[3])) * t)
    return ans

d1_double_exp = jax.jit(jax.jacfwd(double_exp_model_jax))
d2_double_exp = jax.jit(jax.jacfwd(d1_double_exp))
d3_double_exp = jax.jit(jax.jacfwd(d2_double_exp))

double_exp_model_derivs = {
    'd1': d1_double_exp,
    'd2': d2_double_exp,
    'd3': d3_double_exp,
}

@jax.jit
def multi_exp_model_jax(p, t, Nexc=2):
    # p must be entered as p = [A0, E0, A1, E1, ...]
    ans = 0.0
    for i in range(0, Nexc):
        ans += p[2 * i] * jnp.exp(-p[2 * i + 1] * t)
    return ans


def gen_synth_data(
    t, p0, model, frac_noise_amp=1.0, noise_floor_amp=0.0, noise_samples=200):
    """
    Given a model and some numeric parameters (defining the range of independent
    variables x and the 'model truth' p0), generates synthetic data equal to the model
    truth plus some noise.
    Args:
      t: NumPy array specifying the values of independent
         coordinates to sample the model at.
      p0: Dict of the form { par_name: <gv.gvar> } defining the "model truth" values of
          the model parameters.
      model: Model function that takes (x,p0) as inputs - see examples defined above.
      frac_noise_amp: Optional float, determines the magnitude of the noise to be added fractionally.
      noise_floor_amp: Optional float, determines the magnitude of the noise floor.
      noise_samples: Optional int, determines the number of random noise samples
    Returns:
        Dictionary of of synthetic data
    """

    y_exact = model(t, p0)

    frac_noise = np.random.normal(0.0, frac_noise_amp, (noise_samples, len(y_exact)))
    noise_floor = np.random.normal(0.0, noise_floor_amp, (noise_samples, len(y_exact)))
    
    y_noisy = y_exact * (1.0 + frac_noise) + noise_floor

    y = gv.dataset.avg_data(y_noisy)

    return {
        "t": t,
        "yexact": y_exact,
        "y": y,
        "yraw": y_noisy,
        "ND": noise_samples,  # Store sample size in case needed later, e.g. for BIC
    }


def gen_synth_data_corr(
    t,
    p0,
    model,
    rho,
    frac_noise_amp=1.0,
    noise_floor_amp=0.0,
    noise_samples=200,
):
    """
    Given a model and some numeric parameters (defining the range of independent
    variables x and the 'model truth' p0), generates synthetic data equal to the
    model truth plus correlated noise.
    Args:
      t: NumPy array specifying the values of
         independent coordinates to sample the model at.
      p0: Dict of the form { par_name: <gv.gvar> } defining the "model truth" values
          of the model parameters.
      model: Model function that takes (x,p0) as inputs - see examples defined above.
      rho: Correlation coefficient.
      frac_noise_amp: Optional float, determines the magnitude of the noise to be added fractionally.
      noise_floor_amp: Optional float, determines the magnitude of the noise floor.
      noise_samples: Optional int, determines the number of random noise samples to take
    Returns:
        Dictionary of of correlated synthetic data
    """
    assert 0 < rho < 1

    y_exact = model(t, p0)

    # Construct noise array
    Ny = len(y_exact)
    frac_noise_src = gv.gvar([(0, frac_noise_amp)] * Ny)
    frac_noise_corr = np.fromfunction(
        lambda i, j: rho ** (np.abs(i - j)), (Ny, Ny), dtype=np.float64
    )
    frac_noise_src = gv.correlate(frac_noise_src, frac_noise_corr)

    frac_noise_gen = gv.raniter(frac_noise_src)
    frac_noise_array = np.asarray([next(frac_noise_gen) for i in range(noise_samples)])

    noise_floor = np.random.normal(0.0, noise_floor_amp, (noise_samples, len(y_exact)))
    
    y_noisy = y_exact * (1.0 + frac_noise_array) + noise_floor

    y = gv.dataset.avg_data(y_noisy)

    return {
        "t": t,
        "yexact": y_exact,
        "y": y,
        "yraw": y_noisy,
        "ND": noise_samples,  # Store sample size in case needed later, e.g. for BIC
    }


def cut_synth_data_Nsamp(synth_data, Ns_cut):
    """
    Given a synthetic data set, places a cut in the space of random samples, returning
    a reduced data set from the original.  For a cross-validation set, only cuts
    the testing data set, not the training data.
    Args:
      synth_data: A synthetic data set, produced by one of the gen_synth_data functions
        implemented above.
      Ns_cut: How many samples to keep in the cut data set.  Must be less than or equal
        to the number of samples ND in the original synthetic data.
    Returns:
      cut_data: A new synthetic data set formed from the first Ns_cut samples.
    """


    # Check if the synth_data is a list of 2 or just one
    if type(synth_data) is list:
        assert Ns_cut <= synth_data[1]["ND"]

        cut_data = []
        cut_data.append(synth_data[0])  # No cut on training data

        cut_data_raw = synth_data[1]["yraw"][:Ns_cut, :]
        cut_data_avg = gv.dataset.avg_data(cut_data_raw)

        cut_data.append({
            "t": synth_data[1]["t"],
            "yexact": synth_data[1].get("yexact", None),
            "ND": Ns_cut,
            "yraw": cut_data_raw,
            "y": cut_data_avg,
        })
    else:
        assert Ns_cut <= synth_data["ND"]

        cut_data_raw = synth_data["yraw"][:Ns_cut, :]
        cut_data_avg = gv.dataset.avg_data(cut_data_raw)

        cut_data = {
            "t": synth_data["t"],
            "yexact": synth_data.get("yexact", None),
            "ND": Ns_cut,
            "yraw": cut_data_raw,
            "y": cut_data_avg,
        }

    return cut_data