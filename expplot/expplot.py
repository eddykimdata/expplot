# Adapted from https://geoffruddock.com/matplotlib-experiment-visualizations/
import datetime as dt
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform
import logging
import re

plt.rcParams['figure.facecolor'] = 'white'

# logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def calculate_lift(df, lift_type='relative'):
    """
    This function takes in a dataframe with control and exp metric mu, sigma,
    and n, then calculates the lift (relative or absolute) and the standard
    error of the lift. 

    Parameters
    ----------
    df : pd.DataFrame
        Should be in the following format (with the given column names):

        | metric   | dimension   |   mean |   std |    n |   alpha | success_side   |
        |:---------|:------------|-------:|------:|-----:|--------:|:---------------|
        | CTR      | control     |   0.52 |  0.05 | 2500 |    0.05 | positive       |
        | CTR      | A1          |   0.53 |  0.1  | 3500 |    0.05 | positive       |
        | CTR      | A2          |   0.47 |  0.2  | 2000 |    0.05 | positive       |
        | ECTR     | control     |   0.25 |  0.02 | 4000 |    0.05 | negative       |
        | ECTR     | A1          |   0.35 |  0.05 | 2000 |    0.05 | negative       |
        | ECTR     | A2          |   0.33 |  0.15 | 4000 |    0.05 | negative       |

        The columns "success_side" and 'alpha' areoptional.  If not given, the 
        columns will be created with the default success_side value as 'positive'
        and default alpha as 0.05.

    lift_type : str
        Value must be in ['relative', 'absolute'].  Determines whether to return
        the absolute lift (mu_test - mu_control) or relative lift (mu_test - 
        mu_control) / mu_control.  

    Returns
    -------
    df : pd.DataFrame.
        Dataframe with uplift and sem.

        |                |   uplift |        sem | success_side   |
        |:---------------|---------:|-----------:|:---------------|
        | ('CTR', 'A1')  |     0.01 | 0.00196396 | positive       |
        | ('CTR', 'A2')  |    -0.05 | 0.00458258 | positive       |
        | ('ECTR', 'A1') |     0.1  | 0.0011619  | negative       |
        | ('ECTR', 'A2') |     0.08 | 0.0023927  | negative       |
    """
    # Make all column names lower case
    df.columns = map(str.lower, df.columns)
    lift_type = lift_type.lower()
    
    # df must provide required columns 
    for required_column in ['metric', 'dimension', 'mean', 'std', 'n']:
        if required_column not in df.columns:
            raise ValueError("df must provide columns"
                "['metric', 'dimension', mean', 'std', and 'n']")

    # For coloring successful and significant changes as green, define which 
    # side (positive or negative) is the success side. Default to positive.  
    if 'success_side' not in df.columns:
        logger.warn("INFO: df column 'success_side' not found.  Default set to positive side as success.")
        df['success_side'] = 'positive'

    # If alpha is not given, use a default value of 0.05
    if 'alpha' not in df.columns:
        logger.warn("INFO: df column 'alpha' not found.  Default alpha set to 0.05.")
        df['alpha'] = 0.05
    
    # Calculate standard error of the mean (sem)
    df['sem'] = df['std'] / np.sqrt(df['n'])

    # Generate output dataframe
    df_output = pd.DataFrame(index=pd.MultiIndex.from_tuples((),names=['metric', 'dimension']))

    for metric in df['metric'].unique():
        df_metric = df[df['metric']==metric].set_index('dimension')

        # Make sure that the 'control' dimension is present for every metric
        if 'control' not in df_metric.index:
            raise ValueError(
                f"Metric '{metric}' is missing the control group in dimension column.  "
                "Each metric must include a dimension named 'control'")

        for dimension in df_metric.index:
            if dimension=='control':
                continue
            else:
                mu_exp = df_metric.loc[dimension, 'mean']
                mu_ctrl = df_metric.loc['control', 'mean']
                sem_exp = df_metric.loc[dimension, 'sem']
                sem_ctrl = df_metric.loc['control', 'sem']

                if lift_type=='absolute':
                    df_output.loc[(metric, dimension), 'uplift'] = (
                        mu_exp - mu_ctrl)
                    df_output.loc[(metric, dimension), 'sem'] = (
                        np.sqrt(sem_exp**2 + sem_ctrl**2))
                elif lift_type=='relative':
                    df_output.loc[(metric, dimension), 'uplift'] = (
                        (mu_exp - mu_ctrl)/mu_ctrl)
                    # SEM calculated from Delta method 
                    # http://blog.analytics-toolkit.com/2018/confidence-intervals-p-values-percent-change-relative-difference/
                    df_output.loc[(metric, dimension), 'sem'] = (
                        np.sqrt(
                            ((sem_ctrl**2) * mu_exp**2 + (sem_exp**2) * mu_ctrl**2)
                                / (mu_ctrl**4)))

                df_output.loc[(metric, dimension), 'success_side'] = (
                        df_metric.loc[dimension, 'success_side'])
                df_output.loc[(metric, dimension), 'alpha'] = (
                        df_metric.loc[dimension, 'alpha'])

    return df_output


def plot_lift(
    df,
    title=None, 
    combine_axes=False,
    lift_type='relative',
    as_percent=True):
    """
    This function takes in a dataframe with control and exp mean, std, 
    and n, then calculates the lift (relative or absolute) and the standard
    error of the lift. 

    Parameters
    ----------
    df : pd.DataFrame
        Should be in the following format (with the given column names):

        | metric   | dimension   |   mean |   std |    n |   alpha | success_side   |
        |:---------|:------------|-------:|------:|-----:|--------:|:---------------|
        | CTR      | control     |   0.52 |  0.05 | 2500 |    0.05 | positive       |
        | CTR      | A1          |   0.53 |  0.1  | 3500 |    0.05 | positive       |
        | CTR      | A2          |   0.47 |  0.2  | 2000 |    0.05 | positive       |
        | ECTR     | control     |   0.25 |  0.02 | 4000 |    0.05 | negative       |
        | ECTR     | A1          |   0.35 |  0.05 | 2000 |    0.05 | negative       |
        | ECTR     | A2          |   0.33 |  0.15 | 4000 |    0.05 | negative       |

        The columns "success_side" and 'alpha' areoptional.  If not given, the 
        columns will be created with the default success_side value as 'positive'
        and default alpha as 0.05.

    title : str
        Plot title displayed at the top of the plot

    combine_axes : boolean
        Only relevant for multiple-variant experiments.  If false, will plot each
        metric in its own outlined plot.  

    lift_type : str
        Value must be in ['relative', 'absolute'].  Determines whether to return
        the absolute lift (mu_test - mu_control) or relative lift (mu_test - 
        mu_control) / mu_control.  

    as_percent : boolean
        If true, the x-axis will be displayed as percents.

    Returns
    -------
    fig : matplotlib.pyplot.fig

    ax : matplotlib.pyplot.ax
    """

    df_lift = calculate_lift(df, lift_type)
    fig, ax = plot_experiment_results(
        df_lift, 
        title, 
        combine_axes, 
        lift_type, 
        as_percent)

    return fig, ax

def get_experiment_result_colors(
    interval_start: float, 
    interval_end: float,
    success_side='positive',
) -> Tuple[str, str]:

    success_side = success_side.lower()
    if (type(success_side) != str) or (success_side not in ['positive', 'negative']):
        raise ValueError("Column success_side must be in ['positive', 'negative']")
    
    colors = {
        'positive': {
            'positive_fill_color': 'darkseagreen',
            'positive_edge_color': 'darkgreen',
            'negative_fill_color': 'darksalmon',
            'negative_edge_color': 'darkred',
            'neutral_fill_color': 'lightgray',
            'neutral_edge_color': 'gray'},
        'negative': {
            'positive_fill_color': 'darksalmon',
            'positive_edge_color': 'darkred',
            'negative_fill_color': 'darkseagreen',
            'negative_edge_color': 'darkgreen',
            'neutral_fill_color': 'lightgray',
            'neutral_edge_color': 'gray'}
    }

    """ Determine chart colors based on overlap of interval with zero. """
    if interval_start > 0:
        return colors[success_side]['positive_fill_color'], colors[success_side]['positive_edge_color']
    elif interval_end < 0:
        return colors[success_side]['negative_fill_color'], colors[success_side]['negative_edge_color']
    else:
        return colors[success_side]['neutral_fill_color'], colors[success_side]['neutral_edge_color']


def get_xtick_spacing(ax):
    # Get the size of the xtick intervals.  e.g. if xticks are 0.001, 0.002
    # then xtick-precision will be 3 (i.e. 1 * 10^-3). 
    xtick_values = ax.get_xticks()
    xtick_interval = xtick_values[1] - xtick_values[0]
    xtick_interval_sci_not = '{:e}'.format(xtick_interval)

    regex_sci_not = re.search(r'e([\+\-])(\d+)', xtick_interval_sci_not)
    # If interval precision is >= 1 then then do not show decimal places
    if regex_sci_not.group(1)=='+':
        xtick_precision = int(regex_sci_not.group(2))
    else:
        xtick_precision = -int(regex_sci_not.group(2))

    return xtick_values, xtick_interval, xtick_precision


def plot_experiment_single_group(ax, sub_df: pd.DataFrame) -> Tuple[float, float]:
    
    """ Plot each row of a DataFrame on the same mpl axis object. """

    ytick_labels = []
    x_min, x_max = 0, 0

    # Iterate over each row in group, reversing order since mpl plots from bottom up
    for j, (dim, row) in enumerate(sub_df.iloc[::-1].iterrows()):
        if isinstance(dim, tuple):
            dim = dim[1]

        # Calculate z-score for each test based on test-specific correction factor
        z = norm(0, 1).ppf(1 - row.alpha / 2)
        interval_start = row.uplift - (z * row['sem'])
        interval_end = row.uplift + (z * row['sem'])

        # Conditional coloring based on significance of result
        fill_color, edge_color = get_experiment_result_colors(
            interval_start, 
            interval_end,
            success_side=row.success_side
            )

        ax.barh(j, [z * row['sem'], z * row['sem']],
                left=[interval_start, interval_start + z * row['sem']],
                height=0.8,
                color=fill_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=3)

        ytick_labels.append(dim)
        x_min = min(x_min, interval_start)
        x_max = max(x_max, interval_end)

    # Axis-specific formatting

    ax.xaxis.grid(True, alpha=0.4)
    ax.xaxis.set_ticks_position('none')
    ax.axvline(0.00, color='black', linewidth=1.1, zorder=2)
    ax.yaxis.tick_right()
    ax.set_yticks(np.arange(len(sub_df)))
    ax.set_yticklabels(ytick_labels)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min-0.4, y_max+0.4)
    ax.yaxis.set_ticks_position('none')

    return x_min, x_max


def plot_experiment_results(
    df: pd.DataFrame, 
    title: str = None, 
    combine_axes: bool = False,
    lift_type='relative',
    as_percent=True
) -> None:
    """ Plot a (possibly MultiIndex) DataFrame on one or more matplotlib axes.
    Args:
        df (pd.DataFrame): DataFrame with MultiIndex representing dimensions or KPIs, and following cols: uplift, sem, alpha
        title (str): Title displayed above plot
        sample_size (int): Used to add contextual information to bottom corner of plot
        combine_axes (bool): If true and input df has multiindex, collapse axes together into one visible axis.
    """
    lift_type = lift_type.lower()
    plt.rcParams['figure.facecolor'] = 'white'

    # For flipping colors such that negative side is green, the df must have 
    # column named "success_side" with values in ['positive', 'negative'].  
    # If not given, default to always have green bo on the positive side.

    if 'success_side' not in df.columns:
        logger.warn("INFO: df column 'success_side' not found.  Default set to positive side as success.")
        df['success_side'] = 'positive'

    n_levels = len(df.index.names)
    if n_levels > 2:
        raise ValueError
    elif n_levels == 2:
        plt_rows = df.index.get_level_values(0).nunique()
    else:
        plt_rows = 1

    # Make an axis for each group of MultiIndex DataFrame input
    fig, axes = plt.subplots(nrows=plt_rows,
                             ncols=1,
                             sharex=True,
                             figsize=(6, 0.5 * df.shape[0] + 0.2), dpi=100)

    if n_levels == 1:
        ax = axes
        x_min, x_max = plot_experiment_single_group(ax, df)

    if n_levels == 2:
        # Iterate over top-level groupings of index
        x_mins, x_maxs = [], []
        for i, (group, results) in enumerate(df.groupby(level=0, sort=False)):
            ax = axes[i]
            a, b = plot_experiment_single_group(ax, results)
            x_mins.append(a)
            x_maxs.append(b)
            ax.set_ylabel(group)

        x_min = min(x_mins)
        x_max = max(x_maxs)
        ax = axes[-1]  # set variable back to final axis for downstream formatting functions

        if combine_axes:
            fig.subplots_adjust(hspace=0)
            axes[0].spines['bottom'].set_visible(False)
            axes[-1].spines['top'].set_visible(False)
            for axis in axes[1:-1]:
                axis.spines['bottom'].set_visible(False)
                axis.spines['top'].set_visible(False)


    # Get xtick spacing to determine xtick_label display precision
    xtick_values, xtick_interval, xtick_precision = get_xtick_spacing(ax)

    # Create a buffer on xtick min and max, so that bars are not right along
    # the edge of the plot. 
    xtick_buffer = 2 
    x_min = x_min - xtick_buffer*(10**xtick_precision)
    x_max = x_max + xtick_buffer*(10**xtick_precision)
    ax.set_xlim(x_min, x_max)

    # Get new xtick spacing
    xtick_values, xtick_interval, xtick_precision = get_xtick_spacing(ax)

    # Places ticks at sensible intervals. 
    ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=11, steps=[1,2,5,10]))
    
    # Determine how many decimal places to show for each xtick. 
    if (xtick_precision >= 0) & (as_percent==True):
        xtick_format = f"{{:.0%}}"
    elif (xtick_precision >= 0) & (as_percent==False):
        xtick_format = f"{{:.0f}}"
    elif (xtick_precision < 0) & (as_percent==True):
        xtick_format = f"{{:.{max(0, abs(xtick_precision)-2)}%}}"
    elif (xtick_precision < 0) & (as_percent==False):
        xtick_format = f"{{:.{max(0, abs(xtick_precision))}f}}"
    else:
        raise ValueError("as_percent must be boolean [True, False]")

    ax.set_xticklabels([xtick_format.format(x) for x in ax.get_xticks()])
    ax.set_xlabel(f'Uplift ({lift_type})')

    # Add title, sample size, and timestamp labels to plot
    fig.text(0.5, 0.95 - 0.025 * n_levels, title, size='x-large', horizontalalignment='center')

    return fig, ax
