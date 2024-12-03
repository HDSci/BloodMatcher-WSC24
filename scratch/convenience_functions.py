import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


def objectives_stats(file_path) -> list[tuple[float]]:
    """Calculate the mean and standard error of the mean for each objective (column).

        :param str file_path: The path to the objectives TSV file.
        :return list: A list of tuples where each tuple contains the mean and standard error of the mean for a column.
    """
    output_df = pd.read_csv(file_path, sep='\t')
    columns = list(output_df.columns)
    return_value = []
    for column in columns:
        return_value.append((output_df[column].mean(),
                             output_df[column].std() / np.sqrt(len(output_df))))
    return return_value


def objectives_stats_table(files: dict, separate_vals_errs=False) -> tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]] | pd.DataFrame:
    """
    Generates a table of objective statistics from a dictionary of file paths.
    Args:
        files (dict): A dictionary where keys are identifiers and values are file paths.
        separate_vals_errs (bool, optional): If True, returns separate DataFrames for values and errors. 
            Defaults to False.
    Returns:
        tuple[DataFrame, tuple[DataFrame, DataFrame]] | DataFrame: 
            If separate_vals_errs is False, returns a single DataFrame with formatted statistics.
            If separate_vals_errs is True, returns a tuple containing:
                - A DataFrame with formatted statistics.
                - A tuple of two DataFrames: one for values and one for errors.
    """

    data_dict = dict()
    val_dict = dict()
    err_dict = dict()
    for key, value in files.items():
        if value.endswith('_output.tsv'):
            value = value.replace('_output.tsv', '_objectives.tsv')
        value = os.path.realpath(os.path.expanduser(value))
        objstats = objectives_stats(value)
        objstats_str = [f'{x[0]:.3f} ± {x[1]:.4f}' for x in objstats]
        data_dict[key] = objstats_str
        objstats_arr = np.array(objstats)
        val_dict[key] = objstats_arr[:, 0]
        err_dict[key] = objstats_arr[:, 1]
    df = pd.DataFrame(data_dict, index=['alloimmunisations', 'scd_shortages',
                                        'expiries', 'all_shortages', 'O_neg_level', 'O_pos_level', 'O_level',
                                        'D_subs_num_patients', 'ABO_subs_num_patients', 'ABOD_subs_num_patients'])
    if separate_vals_errs:
        df_val = pd.DataFrame(val_dict, index=['alloimmunisations', 'scd_shortages',
                                               'expiries', 'all_shortages', 'O_neg_level', 'O_pos_level', 'O_level',
                                               'D_subs_num_patients', 'ABO_subs_num_patients', 'ABOD_subs_num_patients'])
        df_err = pd.DataFrame(err_dict, index=['alloimmunisations', 'scd_shortages',
                                               'expiries', 'all_shortages', 'O_neg_level', 'O_pos_level', 'O_level',
                                               'D_subs_num_patients', 'ABO_subs_num_patients', 'ABOD_subs_num_patients'])
        return df, (df_val, df_err)
    return df


def read_and_average_comp_times(files: dict) -> pd.DataFrame:
    """
    Reads computation time files, calculates the mean and standard deviation, and returns a DataFrame.

    This function processes a dictionary of file paths, reads the computation times from each file,
    calculates the mean and standard deviation for each set of times, and stores the results in a
    pandas DataFrame.

    Args:
        files (dict): A dictionary where keys are identifiers and values are file paths to the computation
                      time files. The file paths should end with '_output.tsv', which will be replaced
                      with '_computation_times.tsv' to locate the actual computation time files.

    Returns:
        DataFrame: A DataFrame with the mean and standard deviation of computation times for each file.
                   The DataFrame has one row labeled 'computation_time' and columns corresponding to
                   the keys in the input dictionary. The values are formatted as 'mean ± std_dev'.
    """
    data_dict = dict()
    for key, value in files.items():
        if value.endswith('_output.tsv'):
            value = value.replace('_output.tsv', '_computation_times.tsv')
        value = os.path.realpath(os.path.expanduser(value))
        times = np.loadtxt(value, delimiter='\t')
        mean = times.mean()
        std_dev = times.std()
        data_dict[key] = f'{mean:.0f} ± {std_dev:.1f}'
    df = pd.DataFrame(data_dict, index=['computation_time'])
    return df


def exp2_figs(abod, limited, extended, prob_neg_phen=1, figsize=None, labels=['ABOD', 'Limited', 'Extended'],
              colours=['C0', 'C1', 'C2'], ylabels=None, pdfs=None, skip_plots=False):
    """
    Create a series of bar graphs for the expected number of mismatches, substitutions, and alloimmunisations
    for the three matching rules.

    Args:
        abod (str): Path to the ABOD (or the first) file.
        limited (str): Path to the Limited (or the second) file.
        extended (str): Path to the Extended (or the third) file.
        prob_neg_phen (float, optional): Deprecated, fixed to 1. Defaults to 1.
        figsize (tuple, optional): Size of the figure. Defaults to None.
        labels (list, optional): Labels for the three matching rules. 
            Defaults to ['ABOD', 'Limited', 'Extended'].
        colours (list, optional): Colours for the three matching rules. 
            Defaults to ['C0', 'C1', 'C2'].
        ylabels (list, optional): Labels for the y-axis. Defaults to None.
        pdfs (list, optional): Paths to save the figures as PDFs. Defaults to None.
        skip_plots (list, optional): List of booleans to skip plots. Defaults to False.
    Returns:
        None
    """
    abod_df = pd.read_csv(abod, sep='\t', index_col=0)
    limited_df = pd.read_csv(limited, sep='\t', index_col=0)
    extended_df = pd.read_csv(extended, sep='\t', index_col=0)

    labels_mismatch = ["A", "B", "D", "C", "c", "E", "e", "K",
                       "k", "Fya", "Fyb", "Jka", "Jkb", "M", "N", "S", "s"]
    labels_allo = labels_mismatch[3:]
    x1 = np.arange(len(labels_mismatch))
    x2 = np.arange(len(labels_allo))
    width = 0.2

    x = [x2, x2, x2]
    met = ['mismatch', 'subs', 'allo']
    ylabel = ['Number of mismatches', 'Expected number of substitutions',
              'Expected number of alloimmunisations']
    ylabel = ylabels if ylabels is not None else ylabel
    title = ['Mismatches in antigen-negative patients by antigen and matching rule',
             'Substitutions per unit transfused by antigen and matching rule',
             'Alloimmunisations by antigen and matching rule']
    xticks = [labels_allo, labels_allo, labels_allo]
    cols = xticks

    for i in range(len(x)):
        if skip_plots is True or skip_plots[i]:
            continue
        if met[i] == 'subs':
            _neg_phen = 1
        else:
            _neg_phen = prob_neg_phen
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.bar(x[i] - width, abod_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[0],
               yerr=abod_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[0])
        ax.bar(x[i], limited_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[1],
               yerr=limited_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[1])
        ax.bar(x[i] + width, extended_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[2],
               yerr=extended_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[2])
        ax.set_ylabel(ylabel[i])
        ax.set_xticks(x[i])
        ax.set_xticklabels(xticks[i])
        if met[i] == 'subs':
            ax.legend()
        else:
            ax.legend()
        fig.tight_layout()
        plt.show()
        if pdfs is not None and pdfs[i] is not None:
            fig.savefig(pdfs[i], bbox_inches='tight')
            plt.close(fig)


def stacked_stock_levels_graph(datafilename, columns, labels=None, bbox_to_anchor=(1.01, 0.5),
                               xlabel='Time (days)', ylabel='Level of total stock', figsize=(8, 6), dpi=200,
                               demarcate_warmup=False, warmup_color='black', raw_data=False,
                               warmup_period=7*6*3):
    """
    Plots a stacked area graph of stock levels over time from a given data file.

    Parameters:
        datafilename (str): Path to the data file.
            If it ends with '_output.tsv', it will be replaced with '_stocks.tsv'.
        columns (list): List of column names to be plotted.
        labels (list, optional): List of labels for the columns. Defaults to None.
        bbox_to_anchor (tuple, optional): Position of the legend. Defaults to (1.01, 0.5).
        xlabel (str, optional): Label for the x-axis. Defaults to 'Time (days)'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Level of total stock'.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
        dpi (int, optional): Dots per inch for the figure. Defaults to 200.
        demarcate_warmup (bool, optional): Whether to draw a line to demarcate the warmup period. Defaults to False.
        warmup_color (str, optional): Color of the warmup line. Defaults to 'black'.
        raw_data (bool, optional): Whether to use raw data for plotting instead of percentages. Defaults to False.
        warmup_period (int, optional): Length of the warmup period in days. Defaults to 7×6×3 (i.e., 18 weeks).

    Returns:
        None: The function creates and displays a plot.
    """
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_stocks.tsv')
    df = pd.read_csv(os.path.realpath(
        os.path.expanduser(datafilename)), sep='\t')
    if raw_data:
        df = df[columns] * df['total'].values[:, None]
    else:
        df = df[columns]
    if labels is not None:
        df.columns = labels
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.stackplot(np.array(df.index) + 1, df.T.to_numpy(),
                 labels=df.columns, alpha=0.75)
    ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if demarcate_warmup:
        draw_warmup_line(ax, color=warmup_color, x=warmup_period+0.5)
    if not raw_data:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()


def plot_age_dist_blood(datafilename, array='age_dist_given_to_scd', age_range=None,
                        days_range=None, title='Age distribution of blood given to SCD patients over time',
                        vmax=300, cmap='viridis', figsize=(15, 6), dpi=200, re_index=False,
                        print_max=False, every_n_xtick=2, plot_avg=False, print_mean=False,
                        pdf=None) -> pd.DataFrame | None:
    """
    Plots the age distribution of blood given to SCD (Sickle Cell Disease) patients over time.

    Parameters:
        datafilename (str): Path to the data file.
            If it ends with '_output.tsv', it will be replaced with '_age_distributions.npz'.
        array (str): Key to access the specific age distribution data within the .npz file. 
            Determines which graph/data to plot.
            Access the `.files` attribute of the loaded .npz file to see available keys. 
            Default is 'age_dist_given_to_scd'.
        age_range (array-like, optional): Range of ages to include in the plot.
            Default is None, which uses ages 1 to 14.
        days_range (array-like, optional): Range of days to include in the plot.
            Default is None, which uses days 1 to 210.
        title (str): Title of the plot. Default is 'Age distribution of blood given to SCD patients over time'.
        vmax (int): Maximum value for the color scale. Default is 300.
        cmap (str): Colormap to use for the heatmap. Default is 'viridis'.
        figsize (tuple): Size of the figure. Default is (15, 6).
        dpi (int): Dots per inch for the figure. Default is 200.
        re_index (bool): Whether to re-index the days range. Default is False.
        print_max (bool): Whether to print the maximum values in the DataFrame. Default is False.
        every_n_xtick (int): Interval for showing x-tick labels. Default is 2.
        plot_avg (bool): Whether to plot the average age distribution. Default is False.
        print_mean (bool): Whether to print the mean values when plotting the average. Default is False.
        pdf (str, optional): Path to save the plot as a PDF. Default is None.

    Returns:
        out (DataFrame, None): DataFrame of the age distribution if plot_avg is True, otherwise None.
    """
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace(
            '_output.tsv', '_age_distributions.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    age_dist = data[array]
    # Mask the day 0 and day 1 data instead of clipping it out
    age_range = age_range if age_range is not None else np.arange(1, 15)
    days_range = days_range if days_range is not None else np.arange(1, 211)
    days_labels = days_range if not re_index else np.arange(
        len(days_range)) + 1
    arr = age_dist[days_range-1, :]
    arr = arr[:, age_range]
    sba_df = pd.DataFrame(arr.T, index=age_range, columns=days_labels)
    if plot_avg:
        plot_avg_age_dist_blood_scd(sba_df, title=title, ylim_max=vmax,
                                    figsize=figsize, dpi=dpi, pdf=pdf, print_mean=print_mean)
        return sba_df
    # Mask the zero values instead of making them NaNs
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(sba_df.replace(0, np.nan), cmap=cmap, ax=ax,
                cbar_kws={'label': 'Number of units'}, vmax=vmax)
    ax.set_ylabel('Age (days)')
    ax.set_xlabel('Time (days)')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid()
    # Only show every other xtick label
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % every_n_xtick != 0:
            label.set_visible(False)
    fig.tight_layout()
    fig.show()
    data.close()
    if print_max:
        print(sba_df.max(None, skipna=True))
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')


def plot_avg_age_dist_blood_scd(df, title='Mean age distribution of blood given to SCD patients',
                                ylim_max=None, figsize=(8, 6), dpi=200, pdf=None, print_mean=False):
    """
    Plots the average age distribution of blood given to Sickle Cell Disease (SCD) patients.

    Parameters:
        df (pd.DataFrame): DataFrame containing the age distribution data.
        title (str, optional): Title of the plot. Default is 'Mean age distribution of blood given to SCD patients'.
        ylim_max (float, optional): Maximum limit for the y-axis. Default is None.
        figsize (tuple, optional): Size of the figure. Default is (8, 6).
        dpi (int, optional): Dots per inch (DPI) for the figure. Default is 200.
        pdf (str, optional): File path to save the plot as a PDF. Default is None.
        print_mean (bool, optional): If True, prints the mean age of the blood units. Default is False.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    df_hist = df.mean(axis=1)
    i_min = df_hist.index.min()
    i_max = df_hist.index.max()
    ax.stairs(df_hist.to_numpy().flatten(), np.arange(
        i_min, i_max+2) - 0.5, fill=True, color='gray', alpha=1)
    ax.grid()
    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Number of units')
    ax.set_xlim(0, None)
    ax.set_ylim(0, ylim_max)
    ax.set_title(title)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    if print_mean:
        print(np.average(df_hist.index, weights=df_hist))


def ccdf_avg_age_dist_blood_scd(datafilename, array='age_dist_given_to_scd', age_range=None, days_range=None,
                                above_age=0, re_index=False) -> float:
    """
    Calculate the complementary cumulative distribution function (CCDF) of the average age distribution for blood given to SCD patients.

    Parameters:
        datafilename (str): The path to the data file.
            If it ends with '_output.tsv', it will be replaced with '_age_distributions.npz'.
        array (str): The key for the age distribution array in the .npz file. Default is 'age_dist_given_to_scd'.
        age_range (array-like, optional): The range of ages to consider. If None, defaults to np.arange(1, 15).
        days_range (array-like, optional): The range of days to consider. If None, defaults to np.arange(1, 211).
        above_age (int): The age above which to calculate the CCDF. Default is 0.
        re_index (bool): Whether to re-index the days range starting from 1. Default is False.

    Returns:
        float: The CCDF value for the given age distribution and age cut-off.
    """
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace(
            '_output.tsv', '_age_distributions.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    age_dist = data[array]
    age_range = age_range if age_range is not None else np.arange(1, 15)
    days_range = days_range if days_range is not None else np.arange(1, 211)
    days_labels = days_range if not re_index else np.arange(
        len(days_range)) + 1
    arr = age_dist[days_range-1, :]
    arr = arr[:, age_range]
    sba_df = pd.DataFrame(arr.T, index=age_range, columns=days_labels)
    df_hist = sba_df.mean(axis=1)
    i_min = df_hist.index.min()
    i_max = df_hist.index.max()
    tot_mass = df_hist.loc[i_min:i_max].sum()
    mass_above = df_hist[df_hist.index > above_age].sum()
    ccdf = mass_above / tot_mass
    return ccdf


def draw_warmup_line(ax, x=168.5, color='black', linestyle='--', linewidth=2, text="Warm-up Period", xy=None,
                     xytext=None):
    """
    Draws a vertical line on the given Axes object to indicate a warm-up period and annotates it with text.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to draw the line.
        x (float, optional): The x-coordinate for the vertical line. Default is 168.5.
        color (str, optional): The color of the line and text. Default is 'black'.
        linestyle (str, optional): The style of the line (e.g., '--' for dashed). Default is '--'.
        linewidth (int, optional): The width of the line. Default is 2.
        text (str, optional): The text to annotate the line with. Default is "Warm-up Period".
        xy (tuple, optional): The (x, y) coordinates for the annotation.
            Default is None, which calculates the y-coordinate as the mean of the y-axis limits.
        xytext (tuple, optional): The (x, y) coordinates for the text position.
            Default is None, which places the text slightly to the left of the line.

    Returns:
        ax (matplotlib.axes.Axes): The modified Axes object with the warm-up line and annotation.
    """
    y = np.mean(ax.get_ylim())
    if xy is None:
        xy = (x, y)
    if xytext is None:
        xytext = (x * 0.99, y)
    ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.annotate(text, xy=xy, xytext=xytext, rotation='vertical',
                color=color, ha='right', va='center', textcoords='data')
    return ax


def avg_stock_composition(rules, rule_files, donations=None, figsize=(12, 5), dpi=200, ncol=4,
                          bbox_to_anchor=(0.5, -0.12), pdf=None):
    """
    Plots the average stock composition of blood groups based on given rules and rule files.

    Parameters:
        rules (list): List of rule names to be used as labels in the plot.
        rule_files (list): List of file paths containing stock data for each rule.
        donations (ndarray, optional): Array of initial donation proportions for each blood group.
            Defaults to [14.6, 36.2, 2.8, 7.8, 7.8, 28.4, 0.6, 1.8] for O-, O+, B-, B+, A-, A+, AB-, and AB+ respectively.
        figsize (tuple, optional): Size of the figure. Defaults to (12, 5).
        dpi (int, optional): Dots per inch for the figure. Defaults to 200.
        ncol (int, optional): Number of columns in the legend. Defaults to 4.
        bbox_to_anchor (tuple, optional): Bounding box for the legend. Defaults to (0.5, -0.12).
        pdf (str, optional): File path to save the figure as a PDF. If None, the figure is not saved. Defaults to None.

    Returns:
        None
    """
    blood_grps = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']
    if donations is None:
        donations = np.array([14.6, 36.2, 2.8, 7.8, 7.8, 28.4, 0.6, 1.8])
    data = [donations / donations.sum()]
    files = []
    for rule in rule_files:
        stock = pd.read_csv(rule.replace(
            '_output.tsv', '_stocks.tsv'), sep='\t')
        stock = stock[blood_grps].to_numpy()[-42:].mean(axis=0)
        data.append(stock / stock.sum())
    data = np.transpose(data)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(data.shape[0]):
        ax.bar(rules, data[i], bottom=np.sum(
            data[:i], axis=0), label=blood_grps[i])
    ax.set_ylabel('Proportion of Total Stock')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, loc='upper center')
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    plt.show()
