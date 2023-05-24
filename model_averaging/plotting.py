#!/usr/bin/env python3

import matplotlib.pyplot as plt
import gvar as gv
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

from IPython.display import set_matplotlib_formats

set_matplotlib_formats("png", "pdf")

# Settings for "publication-ready" figures
color_palette = sns.color_palette("deep")
sns.set_palette(color_palette)
sns.palplot(color_palette)

sns.set(style="white")
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

sns.set_context(
    "paper", font_scale=2.0, rc={"lines.linewidth": 2.5, "figure.figsize": (7, 5)}
)

# Color, marker, and line conventions for IC comparison plots
color_blind = sns.color_palette('colorblind')
IC_color = {'BAIC_perf': color_blind[0],
            'BAIC_sub': color_blind[3],
            'indv': color_blind[4],
            'indv_1': color_blind[6],
            'naive': color_blind[7],
           }

IC_marker = {'BAIC_perf': 'D',
             'BAIC_sub': 's',
             'indv': 'o',
             'naive': '^',
            }


IC_linestyle = {'BAIC_perf': '-',
                'BAIC_sub': '--',
                'naive': ':',
               }

def plot_gvcorr(
    gc,
    color="blue",
    log_scale=False,
    offset=0.0,
    x=None,
    xr_offset=True,
    label=None,
    marker="o",
    markersize=6,
    capthick=2,
    capsize=4,
    open_symbol=False,
    fill=False,
    linestyle=" ",
    eb_linestyle=" ",
):
    if x is None:
        x = np.arange(0, len(gc))

    if fill:
        y = np.asarray([gv.mean(g) for g in gc])
        yerr = np.asarray([gv.sdev(g) for g in gc])
        eplot = plt.plot(x + offset, y, color=color, label=label)
        eplot = plt.fill_between(
            x + offset, y - yerr, y + yerr, alpha=0.3, edgecolor="k", facecolor=color
        )
    else:
        if open_symbol:
            eplot = plt.errorbar(
                x=x + offset,
                y=[gv.mean(g) for g in gc],
                yerr=[gv.sdev(g) for g in gc],
                marker=marker,
                markersize=markersize,
                capthick=capthick,
                capsize=capsize,
                linestyle=linestyle,
                color=color,
                label=label,
                mfc="None",
                mec=color,
                mew=1,
            )
        else:
            eplot = plt.errorbar(
                x=x + offset,
                y=[gv.mean(g) for g in gc],
                yerr=[gv.sdev(g) for g in gc],
                marker=marker,
                markersize=markersize,
                capthick=capthick,
                capsize=capsize,
                linestyle=linestyle,
                color=color,
                label=label,
            )

        if eb_linestyle != " ":
            eplot[-1][0].set_linestyle(eb_linestyle)

    if log_scale:
        plt.yscale("symlog")

    if xr_offset:
        plt.xlim(x[0] - 0.2, x[-1] + 0.2)

    return eplot

def plot_MA_result(test_res, IC_list=['BAIC_perf', 'BAIC_BMW'], target_value=None, show_IC_ratios=False, indv_plot_key='indv', ylim=None, xlim=None):
    """
    Creates plot of model average results.
    Args:
      test_res: Dictionary of model average test results.
      IC_list: list of information criteria to be plotted.
      target_value: Value of model truth
      show_IC_ratios: Boolean flag, default false. If true, plots third panel of ratio of model probabilities.
      indv_plot_key: Used to specify color of individual fit results
      ylim: y-axis limits for first panel of plot
      xlim: x-axis limits for all panels of plot
    """
    
    if show_IC_ratios:
        fig = plt.figure(figsize=(7,6))

        gs = plt.GridSpec(3, 1, height_ratios=[3,1,1])
        gs.update(hspace=0.26)
    else:
        fig = plt.figure(figsize=(7,5))

        gs = plt.GridSpec(2, 1, height_ratios=[3,1])
        gs.update(hspace=0.16)

    ax1 = plt.subplot(gs[0])
    
    y_label = r'$a_0$'
    x_label = r'$t_{\rm min}$'
    x_coordn = 'tmin'
    IC_x_coordn_start = 1.5
    IC_x_coordn_space = 1.0
    x_tick_spacing = 4
        
    IC_x_coordn = IC_x_coordn_start
    for IC in IC_list:
        plot_gvcorr([test_res['obs_avg_IC'][IC]], x=np.array([IC_x_coordn]), color=IC_color[IC], markersize=6, marker=IC_marker[IC], open_symbol=False, label='Model avg. ('+IC+')')
        IC_x_coordn = IC_x_coordn + IC_x_coordn_space

    plot_gvcorr(test_res['obs'], x=test_res[x_coordn], color=IC_color[indv_plot_key], markersize=6, marker=IC_marker['indv'], label='Individual fits')

    if ylim is None:
        y_center = gv.mean(test_res['obs_avg_IC'][IC_list[0]])
        ylim = [0.5*y_center,1.5*y_center]
    if xlim is None:
        xlim = [test_res[x_coordn][0]-0.3,test_res[x_coordn][-1]+0.3]

    plt.yticks(np.arange(ylim[0], ylim[1]+1, step=0.4))
        
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
        
    ax1.set_ylabel(y_label)
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_ylim(ylim[0],ylim[1])
    plt.setp(ax1.get_xticklabels(), visible=False)

    if target_value is not None:
        plt.axhline(target_value, color='k', linestyle='--', lw=2)

    ax2 = plt.subplot(gs[1])

    p_norm = {IC: test_res['prob'][IC] / np.sum(test_res['prob'][IC]) for IC in IC_list}
    Q_norm = test_res['Qs']

    for IC in IC_list:
        plt.plot(test_res[x_coordn], p_norm[IC], color=IC_color[IC], linestyle=IC_linestyle[IC], label='pr$(M|D)$ ('+IC+')')

    ax2r = ax2.twinx()
    ax2r.plot(test_res[x_coordn], np.asarray(Q_norm), color=IC_color['naive'], linestyle=IC_linestyle['naive'], label='Fit $p$-value')  # Note: fit prob != model prob! 

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
    plt.yticks([0,np.max(p_norm[IC_list[0]])])
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else '{:.2f}'.format(x)))

    ax2.set_ylabel(r'${\rm pr}$')
    ax2.set_xlim(xlim[0],xlim[1])

    ax2r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else '{:.1f}'.format(x)))
    ax2r.set_ylabel('Q')
    ax2r.set_yticks([0,1])


    if show_IC_ratios:
        ax2.set_xticklabels([])
        
        ax3 = plt.subplot(gs[2])

        ax3.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))

        for IC in IC_list[1:]:
#             plt.plot(test_res[x_coordn], p_norm[IC] / p_norm[IC_list[0]], color=IC_color[IC], linestyle=IC_linestyle[IC], label='pr$(M|D)$ ('+IC+')/pr$(M|D)$ ('+IC_list[0]+')')
            plt.plot(test_res[x_coordn], p_norm[IC] / p_norm[IC_list[0]], color='k', linestyle='-', label='pr$(M|D)$ ('+IC+')/pr$(M|D)$ ('+IC_list[0]+')')

        ax3.set_xlabel(x_label)
#         ax3.set_ylabel(r'$pr / pr_{\rm '+IC_list[0]+'}$')
        ax3.set_ylabel(r'r')
        ax3.set_xlim(xlim[0],xlim[1])
    else:
        ax2.set_xlabel(x_label)
        
         
        
def plot_MA_result_scaling(obs_est_vs_Nsamp, Nsamp_array, IC_list=['BAIC', 'BPIC', 'PPIC'], fixed_list=[], target_value=None, indv_plot_key='indv', ylim=None, xlim=None):
    """
    Creates N-scaling plot of model average results.
    Args:
      obs_est_vs_Nsamp: Dictionary of IC and fixed-model parameter estimates versus the number of samples N
      Nsamp_array: Array of the number of samples N
      IC_list: list of information criteria to be plotted.
      fixed_list: list of fixed models to be plotted.
      target_value: Value of model truth
      indv_plot_key: Used to specify color of individual fit results
      ylim: y-axis limits for first panel of plot
      xlim: x-axis limits for all panels of plot
    Returns:
      ax: matplotlib axis object for the figure
    """
    
    fig = plt.figure(figsize=(7,5))
    gs = plt.GridSpec(1, 1, height_ratios=[3])
    gs.update(hspace=0.16)

    ax = plt.subplot(gs[0])
    
    y_label = r'$a_0$'
    IC_x_coordn_start = 0.08
    IC_x_coordn_space = 0.10

    IC_x_coordn = IC_x_coordn_start
    for IC in IC_list:
        plot_gvcorr(obs_est_vs_Nsamp[IC], x=np.log(Nsamp_array) + IC_x_coordn, color=IC_color[IC], marker=IC_marker[IC], markersize=6, label='Model avg. ('+IC+')')
        IC_x_coordn = IC_x_coordn + IC_x_coordn_space
    for fixed in fixed_list:
        plot_gvcorr(obs_est_vs_Nsamp[fixed], x=np.log(Nsamp_array) + IC_x_coordn, color=IC_color[indv_plot_key], marker=IC_marker['indv'], markersize=6)
        IC_x_coordn = IC_x_coordn + IC_x_coordn_space
        
    if target_value is not None:
        plt.axhline(target_value, color='k', linestyle='--', lw=2)

    if ylim is None:
        y_center = gv.mean(obs_est_vs_Nsamp[IC_list[0]][-1])
        ylim = [0.5*y_center,1.5*y_center]
    if xlim is None:
        xlim = [np.log(Nsamp_array[0])-0.2,np.log(Nsamp_array[-1])+0.85]
        
    plt.yticks(np.arange(ylim[0], ylim[1]+1, step=0.4))
        
    ax.set_xlabel(r'$\log(N)$')
    ax.set_ylabel(y_label)
    ax.set_xlim(xlim[0],xlim[-1])
    ax.set_ylim(ylim[0],ylim[-1])
    
    return ax
