#!/usr/bin/env python3

from .synth_data import *
import lsqfit

def run_fit_poly(
    data, Nt, m, prior_width=10, **kwargs,
):
    """
    Setup and run a fit against a polynomial model of order m.
    Args:
      data: Synthetic data set to run fits against.
      Nt: Normalizing constant for the polynomial model
          (typically Nt=maximum value of the independent coordinate.)
      m: Order of the polynomial model to fit.
      prior_width: Width of priors for model fit parameters (default: 10).
    Returns:
      fr: an lsqfit FitResults object.
      design_mat: the design matrix for the fit.
    kwargs are passed to the lsqfit.nonlinear_fit function.
    """

    priors_poly = {}
    for i in range(0, m + 1):
        priors_poly[f"a{i}"] = gv.gvar(0.0, prior_width)

    def fit_model(x, p):
        return poly_model_lsqfit(x, p, Nt=Nt, m=m)

    fr = lsqfit.nonlinear_fit(
        data=(data["t"], data["y"]),
        fcn=fit_model,
        prior=priors_poly,
        **kwargs
    )
    
    def jax_model(p, x):
        return poly_model_jax(p, x, Nt=Nt, m=m)
    
    d1_model = lambda p,x: jax.jacfwd(jax_model)(p,x)
    design_mat = d1_model(gv.mean(fr.p.values()),data["t"])

    return fr, design_mat

def run_fit_single_exp(
    data,
    Nt=32,
    priors_SE=None,
    fr_alt_guess=None,
    **kwargs,
):
    """
    Setup and run a fit against the single-exponential model.
    Args:
      data: Synthetic data set to run fits against.
      Nt: "Finite volume" parameter for the exponential model
      priors_SE: dictionary of priors.  Overrides the default priors
                 used if equal to None.
      fr_alt_guess: an lsqfit FitResults object containing an alternate
                    initial guess for E0 if the fitting fails to
                    converge (e.g., the result from a previous fit)
    Returns:
      fr: an lsqfit FitResults object.
      model_derivs: disctionary of the first three derivative tensors
                    of the model function evaluated at the best-fit
                    parameter
    kwargs are passed to the lsqfit.nonlinear_fit function.
    """

    if priors_SE is None:
        priors_SE = {
            "A0": gv.gvar("0(10)"),
            "E0": gv.gvar("1(1)"),
        }

    fr = lsqfit.nonlinear_fit(
        data=(data["t"], data["y"]),
        fcn=single_exp_model_lsqfit,
        prior=priors_SE,
        **kwargs
    )
    
    # If the fit failed to converge, try again with different initial guess based on inputed fit object
    if fr_alt_guess is not None:
        if fr.psdev["E0"] > priors_SE["E0"].sdev / 2:

            p0 = {"A0": fr_alt_guess.pmean["A0"],
                  "E0": fr_alt_guess.pmean["E0"],
                 }
        
            fr = lsqfit.nonlinear_fit(
                data=(data["t"], data["y"]),
                fcn=single_exp_model_lsqfit,
                prior=priors_SE,
                p0=p0,
                **kwargs
            )
                
    # Must match ordering as specified in synth_data.py
    jax_pars = ["A0", "E0"]

    model_derivs = compute_derivatives(single_exp_model_derivs,fr,data,jax_pars)
    
    return fr, model_derivs

def run_fit_double_exp(
    data,
    Nt=32,
    priors_SE=None,
    fr_alt_guess=None,
    **kwargs,
):
    """
    Setup and run a fit against a two-exponential model.
    Args:
      data: Synthetic data set to run fits against.
      Nt: "Finite volume" parameter for the exponential model
      priors_SE: dictionary of priors.  Overrides the default priors
                 used if equal to None.
      fr_alt_guess: an lsqfit FitResults object containing an alternate
                    initial guess for E0 if the fitting fails to
                    converge (e.g., the result from a previous fit)
    Returns:
      fr: an lsqfit FitResults object.
      model_derivs: disctionary of the first three derivative tensors
                    of the model function evaluated at the best-fit
                    parameter
    kwargs are passed to the lsqfit.nonlinear_fit function.
    """

    if priors_SE is None:
        priors_SE = {
            "A0": gv.gvar("0(10)"),
            "E0": gv.gvar("0.4(4)"),
            "A1": gv.gvar("0(10)"),
            "ldE1": gv.gvar("-1(3)"),
        }

    fr = lsqfit.nonlinear_fit(
        data=(data["t"], data["y"]),
        fcn=double_exp_model_lsqfit,
        prior=priors_SE,
        tol=(1e-14,1e-14,1e-14),
        **kwargs
    )
    
    # If the fit failed to converge, try again with different initial guess based on inputed fit object
    if fr_alt_guess is not None:
        if fr.psdev["E0"] > priors_SE["E0"].sdev / 2:

            p0 = {"A0": fr_alt_guess.pmean["A0"],
                  "E0": fr_alt_guess.pmean["E0"],
                  "A1": fr_alt_guess.pmean["A1"],
                  "ldE1": fr_alt_guess.pmean["ldE1"],
                 }
        
            fr = lsqfit.nonlinear_fit(
                data=(data["t"], data["y"]),
                fcn=single_exp_model_lsqfit,
                prior=priors_SE,
                p0=p0,
                **kwargs
            )
                
    # Must match ordering as specified in synth_data.py
    jax_pars = ["A0", "E0", "A1", "ldE1"]

    model_derivs = compute_derivatives(double_exp_model_derivs,fr,data,jax_pars)
    
    return fr, model_derivs


deriv_cache = {}

def compute_derivatives(model_derivs, fit, data, jax_pars):
    """
    compute the first three derivative tensors of the model function
      evaluated at the best-fit parameter.
    Args:
      model_derivs: model derivatives compued with jax in from synth_data.py
      fit: an lsqfit FitResults object.
      data: Data with which to evaluate the derivatives.
      jax_pars: List of parameter names.
    Returns:
      model_derivs_bf: disctionary of the first three derivative tensors
                       of the model function evaluated at the best-fit
                       parameter
    """
    
    ## v1: no JIT
    # d1_model = lambda p,t: jax.jacfwd(model_fcn)(p,t)
    # d2_model = lambda p,t: jax.jacfwd(d1_model)(p,t)
    # d3_model = lambda p,t: jax.jacfwd(d2_model)(p,t)

    ## v2: inline JIT, but not persistent between runs
    # d1_model = jax.jit(jax.jacfwd(model_fcn))
    # d2_model = jax.jit(jax.jacfwd(d1_model))
    # d3_model = jax.jit(jax.jacfwd(d2_model))

    ## v3: persistent JIT, defined in synth_data.py
    d1_model = model_derivs['d1']
    d2_model = model_derivs['d2']
    d3_model = model_derivs['d3']

    bf_p = gv.mean([ fit.p[j] for j in jax_pars])
    
    d1_model_best_fit = d1_model(bf_p,data["t"])
    d2_model_best_fit = d2_model(bf_p,data["t"])
    d3_model_best_fit = d3_model(bf_p,data["t"])


    model_derivs_bf = {'d1_model': d1_model_best_fit, 'd2_model': d2_model_best_fit, 'd3_model': d3_model_best_fit}
    
    return model_derivs_bf