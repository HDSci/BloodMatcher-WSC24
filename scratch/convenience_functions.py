import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick


def alloimmunisation_stats(file_path):
    output_df = pd.read_csv(file_path, sep='\t', index_col=0)
    allo_stat = output_df.loc['allo_avg', :].sum(), np.sqrt(
        ((output_df.loc['allo_stderr', :]).to_numpy()**2).sum())
    return allo_stat


def stats(file_path, stat='allo'):
    output_df = pd.read_csv(file_path, sep='\t', index_col=0)
    out_stat = output_df.loc[f'{stat}_avg', :].sum(), np.sqrt(
        ((output_df.loc[f'{stat}_stderr', :]).to_numpy()**2).sum())
    return out_stat


def objectives_stats(file_path):
    output_df = pd.read_csv(file_path, sep='\t')
    columns = list(output_df.columns)
    return_value = []
    for column in columns:
        return_value.append((output_df[column].mean(),
                             output_df[column].std() / np.sqrt(len(output_df))))
    # alloimmunisations = output_df['alloimmunisations'].mean(), output_df['alloimmunisations'].std() / np.sqrt(len(output_df))
    # shortages = output_df['shortages'].mean(), output_df['shortages'].std() / np.sqrt(len(output_df))
    # expirys = output_df['expirys'].mean(), output_df['expirys'].std() / np.sqrt(len(output_df))
    # return alloimmunisations, shortages, expirys
    return return_value


def objectives_stats_table(files: dict, separate_vals_errs=False):
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


def read_and_average_comp_times(files: dict):
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


def objectives_stats_comparison_chart(files: dict, baseline: str, figsize=(12, 8), dpi=200, rows=None, cols=None):
    _, (df_val, df_err) = objectives_stats_table(
        files, separate_vals_errs=True)

    # Calculate percent change from baseline for df_val and df_err
    df_val_pct_change = df_val.div(df_val[baseline], axis=0) - 1
    # df_err_pct_change = df_err.div(df_err[baseline], axis=0) - 1

    # Calculate the error propagation
    df_err_pct_change = (df_val_pct_change + 1) * ((df_err / df_val)
                                                   ** 2 + (df_err[baseline] / df_val[baseline]) ** 2) ** 0.5

    # Drop the baseline column
    df_val_pct_change = df_val_pct_change.fillna(0)
    df_err_pct_change = df_err_pct_change.fillna(0)
    df_val_pct_change = df_val_pct_change.drop(baseline, axis=1)
    df_err_pct_change = df_err_pct_change.drop(baseline, axis=1)

    # Transpose the dataframes for plotting
    df_val_pct_change = df_val_pct_change.transpose()
    df_err_pct_change = df_err_pct_change.transpose()

    # Create the plot
    # fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # df_val_pct_change.plot(kind='bar', yerr=df_err_pct_change, ax=ax)
    # ax.set_ylabel('Percent Change in Objective Value')
    # ax.set_xlabel('Objective')
    # ax.set_title('Percent Change in Objective Values Relative to Baseline')
    # ax.legend(bbox_to_anchor=(1.0, 1.0))
    # fig.tight_layout()

    # return fig, ax

    # Create a color palette
    colours = sns.color_palette('deep', df_val_pct_change.shape[1])

    # Create a subplot for each objective
    rows = df_val_pct_change.shape[1] if rows is None else rows
    cols = 1 if cols is None else cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    for i, (objective, ax) in enumerate(zip(df_val_pct_change.columns, axs.flatten())):
        df_val_pct_change[objective].plot(
            kind='bar', yerr=df_err_pct_change[objective], ax=ax, color=colours[i])
        ax.set_ylabel('Percent Change')
        # ax.set_xlabel('Scenario')
        ax.grid(axis='y')
        ax.set_title(f'Percent Change in {objective} Relative to Baseline')

        # Set the x-label only for the last subplots
        if i/cols >= rows - 1:
            ax.set_xlabel('Objective')
        else:
            ax.set_xticklabels([])

        # Format the y-axis as percentages
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.tight_layout()

    return fig, axs

# def objectives_stats_comparison_chart(files: dict, baseline: str, figsize=(12, 8), dpi=200):
#     df = objectives_stats_table(files)


#     baseline_stats = df1[baseline]
#     df1 = df1.drop(baseline, axis=1)
#     df1 = df1.astype(float)
#     df1 = df1.subtract(baseline_stats, axis=0)
#     df1 = df1.transpose()
#     df1 = df1.sort_values(by=['alloimmunisations', 'scd_shortages', 'expiries', 'all_shortages'], ascending=False)
#     df1 = df1.transpose()
#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
#     df1.plot(kind='bar', ax=ax)
#     ax.set_ylabel('Change in objective value')
#     ax.set_xlabel('Objective')
#     ax.set_title('Change in objective values relative to baseline')
#     ax.legend(bbox_to_anchor=(1.0, 1.0))
#     fig.tight_layout()
#     return fig, ax


def antigens():
    from BSCSimulator.antigen import Antigens
    from BSCSimulator.experiments.allo_incidence import (ANTIGENS,
                                                         abd_usability,
                                                         load_alloantibodies,
                                                         load_immunogenicity,
                                                         load_rule_sets)

    matching_antigens = load_rule_sets(
        'BSCSimulator/experiments/matching_rules.json')['MATCHING_RULES']['Extended']['antigen_set']
    alloantibodies = load_alloantibodies()
    immuno = load_immunogenicity()

    Antigens.population_abd_usabilities = abd_usability()
    antigens = Antigens(ANTIGENS, rule=matching_antigens,
                        allo_Abs=alloantibodies.flatten())
    antigens.allo_risk = immuno[antigens.antigen_index[3:]].to_numpy(
    ).flatten()
    return antigens


def exp2_figs(abod, limited, extended, prob_neg_phen=1, figsize=None, labels=['ABOD', 'Limited', 'Extended'],
              colours=['C0', 'C1', 'C2'], ylabels=None, pdfs=None, skip_plots=False):
    """Figures for Experiment 2

    :param abod:
    :param limited:
    :param extended:
    :param prob_neg_phen:
    :param figsize:
    :return:
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
    # xticks = [labels_mismatch, labels_mismatch, labels_allo]
    xticks = [labels_allo, labels_allo, labels_allo]
    cols = xticks

    for i in range(len(x)):
        if skip_plots is True or skip_plots[i]:
            continue
        if met[i] == 'subs':
            _neg_phen = 1
        else:
            _neg_phen = prob_neg_phen
        # fig, ax = plt.subplots(figsize=(12, 8))
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.bar(x[i] - width, abod_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[0],
               yerr=abod_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[0])
        ax.bar(x[i], limited_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[1],
               yerr=limited_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[1])
        ax.bar(x[i] + width, extended_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[2],
               yerr=extended_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[2])
        # ax.set_ylabel(ylabel[i], fontsize=14)
        ax.set_ylabel(ylabel[i])
        # ax.set_title(title[i], fontsize=18)
        # ax.set_title(title[i])
        ax.set_xticks(x[i])
        # ax.set_xticklabels(xticks[i], fontsize=14)
        ax.set_xticklabels(xticks[i])
        if met[i] == 'subs':
            ax.legend()
        else:
            ax.legend()
            # ax.legend(bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        plt.show()
        if pdfs is not None and pdfs[i] is not None:
            fig.savefig(pdfs[i], bbox_inches='tight')
            plt.close(fig)


def exp3_figs(limited, extended, prob_neg_phen=1, figsize=(12, 8), labels=['Limited', 'Extended']):

    limited_df = pd.read_csv(limited, sep='\t', index_col=0)
    extended_df = pd.read_csv(extended, sep='\t', index_col=0)

    labels_mismatch = ["A", "B", "D", "C", "c", "E", "e", "K",
                       "k", "Fya", "Fyb", "Jka", "Jkb", "M", "N", "S", "s"]
    labels_allo = labels_mismatch[3:]
    x1 = np.arange(len(labels_mismatch))
    x2 = np.arange(len(labels_allo))
    width = 0.35

    x = [x2, x2, x2]
    met = ['mismatch', 'subs', 'allo']
    ylabel = ['Number of mismatches', 'Expected number of substitutions',
              'Expected number of alloimmunisations']
    title = ['Mismatches in antigen-negative patients by antigen and matching rule', 'Substitutions per unit transfused by antigen and matching rule',
             'Alloimmunisations by antigen and matching rule']
    # xticks = [labels_mismatch, labels_mismatch, labels_allo]
    xticks = [labels_allo, labels_allo, labels_allo]
    cols = xticks

    for i in range(len(x)):
        if met[i] == 'subs':
            _neg_phen = 1
        else:
            _neg_phen = prob_neg_phen
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.bar(x[i] - width/2, limited_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[0],
               yerr=limited_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color='tab:blue')
        ax.bar(x[i] + width/2, extended_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[1],
               yerr=extended_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color='tab:green')
        # ax.set_ylabel(ylabel[i], fontsize=14)
        ax.set_ylabel(ylabel[i])
        # ax.set_title(title[i], fontsize=18)
        # ax.set_title(title[i])
        ax.set_xticks(x[i])
        # ax.set_xticklabels(xticks[i], fontsize=14)
        ax.set_xticklabels(xticks[i])
        ax.legend()
        fig.tight_layout()
        plt.show()


def create_histogram_panels(data1, data2, data3):
    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)

    # Plot histogram for dataset 1
    axes[0].hist(data1, bins=20, color='blue', alpha=0.5)
    axes[0].set_title('Limited Rule')

    # Plot histogram for dataset 2
    axes[1].hist(data2, bins=20, color='green', alpha=0.5)
    axes[1].set_title('Extended Rule')

    # Plot histogram for dataset 3
    axes[2].hist(data3, bins=20, color='red', alpha=0.5)
    axes[2].set_title('Optimised Extended Rule')

    # Label axes
    for ax in axes:
        ax.set_ylabel('Frequency')
    axes[2].set_xlabel('Total expected alloimmunisations')

    # Adjust spacing between subplots
    fig.tight_layout()

    # Display the chart
    plt.show()


def create_histograms(data1, data2, data3, xlabel='Total expected alloimmunisations', title='Distribution of total expected alloimmunisations',
                      labels=['Limited Rule', 'Extended Rule', 'Extended Anticipation Rule'], density=False, xlim=None):
    # Create a figure and axes for the panel
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Plot histograms for all datasets
    # ax.hist(data1, color='tab:blue', alpha=0.5, label=labels[0], density=density)
    # ax.hist(data2, color='tab:green', alpha=0.5, label=labels[1], density=density)
    # ax.hist(data3, color='tab:red', alpha=0.5, label=labels[2], density=density)

    # Using seaborn.kdeplot instead
    # sns.kdeplot(data1, color='tab:blue', alpha=0.5, label='Limited Rule', ax=ax, shade=True)
    # sns.kdeplot(data2, color='tab:green', alpha=0.5, label='Extended Rule', ax=ax, fill=True)
    # sns.kdeplot(data3, color='darkgreen', alpha=0.5, label='Optimised Extended Rule', ax=ax, fill=True)

    # Using seaborn.histplot instead
    sns.histplot(data1, color='tab:blue', alpha=0.5,
                 label=labels[0], ax=ax, kde=density)
    sns.histplot(data2, color='tab:green', alpha=0.5,
                 label=labels[1], ax=ax, kde=density)
    sns.histplot(data3, color='tab:red', alpha=0.5,
                 label=labels[2], ax=ax, kde=density)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    # ax.set_xlim(0, 12)

    # Add legend
    ax.legend()

    fig.tight_layout()

    # Display the chart
    plt.show()


def get_data_from_file_for_histograms(filenames: list, data_type: str):
    """
    Get data from files for histograms
    :param filenames: list of filenames
    :param data_type: column name in the file
    :return: tuple of arrays of data
    """
    data = []
    for filename in filenames:
        if filename.endswith('_output.tsv'):
            filename = filename.replace('_output.tsv', '_objectives.tsv')
        df = pd.read_csv(os.path.realpath(
            os.path.expanduser(filename)), sep='\t')
        data.append(df[data_type].to_numpy())
    return tuple(data)


def create_stock_levels_graph(datafilename, columns, labels=None, xlabel='Time (days)',
                              ylabel='Number of units', figsize=(8, 6), dpi=300,
                              ylim=None, xlim=None, raw_data=False):
    """
    Create a graph of stock levels
    :param datafilename: name of the file with data
    :param columns: list of columns to plot
    :param labels: list of labels for columns
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param figsize: size of the figure
    :param dpi: resolution of the figure
    :param ylim: tuple of (lower, upper) bounds for the y-axis
    :return: None
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
    fig = plt.figure(dpi=dpi)
    ax = fig.gca()
    # ax.grid(True)
    ax = df.plot(figsize=figsize, kind='line', xlabel=xlabel,
                 ylabel=ylabel, ax=ax, grid=True)
    if not raw_data:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)


def stacked_stock_levels_graph(datafilename, columns, labels=None, bbox_to_anchor=(1.01, 0.5),
                               xlabel='Time (days)', ylabel='Level of total stock', figsize=(8, 6), dpi=200,
                               demarcate_warmup=False, warmup_color='black', raw_data=False,
                               warmup_period=7*6*3):
    """
    Create a stacked graph of stock levels
    :param datafilename: name of the file with data
    :param columns: list of columns to plot
    :param labels: list of labels for columns
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param figsize: size of the figure
    :param dpi: resolution of the figure
    :return: None
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
                        pdf=None):
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
    # fig.show()
    if print_mean:
        print(np.average(df_hist.index, weights=df_hist))


def ccdf_avg_age_dist_blood_scd(datafilename, array='age_dist_given_to_scd', age_range=None,
                                days_range=None, above_age=0, re_index=False):
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
    # df_hist = df_hist.sort_index(ascending=False)
    # ccdf = df_hist.cumsum()
    # ccdf = ccdf / ccdf.max()
    ccdf = mass_above / tot_mass
    return ccdf


def draw_warmup_line(ax, x=168.5, color='black', linestyle='--', linewidth=2, text="Warm-up Period",
                     xy=None, xytext=None):
    y = np.mean(ax.get_ylim())
    if xy is None:
        xy = (x, y)
    if xytext is None:
        xytext = (x * 0.99, y)
    ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.annotate(text, xy=xy, xytext=xytext, rotation='vertical',
                color=color, ha='right', va='center', textcoords='data')
    return ax


def reshape_abo_combos(datafilename, mixed_match=False, stderr=0):
    if mixed_match:
        suffix = '_abodmm_subs.tsv'
    else:
        suffix = '_abocm.tsv'
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', suffix)
    labels = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']
    data = pd.read_csv(os.path.realpath(os.path.expanduser(
        datafilename)), sep='\t', nrows=2, index_col=0)
    return pd.DataFrame(data.values[stderr, :].reshape(8, 8), columns=labels, index=labels)


def add_totals_to_abo_combos(reshaped_abo_combos):
    reshaped_abo_combos['Total'] = reshaped_abo_combos.sum(axis=1)
    reshaped_abo_combos.loc['Total'] = reshaped_abo_combos.sum(axis=0)
    return reshaped_abo_combos


def mixed_allocation_totals(datafilename, print_with_stderr=False, breakdowns=False):
    # reshaped_means = reshape_abo_combos(datafilename, mixed_match=True)
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_abodmm_subs.tsv')
    data = pd.read_csv(os.path.realpath(
        os.path.expanduser(datafilename)), sep='\t', nrows=2, index_col=0)
    abo_x = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    abo_mm_indices = abo_x > abo_x[:, None]
    d_x = np.array([0, 1] * 4)
    d_mm_indices = d_x > d_x[:, None]
    abod_mm_indices = abo_mm_indices & d_mm_indices
    indices = (d_mm_indices, abo_mm_indices, abod_mm_indices)
    compatible_indices = (
        [True, True, True, True,                # O-
         False, True, False, True,              # B-
         False, False, True, True,              # A-
         False, False, False, True],            # AB-
        [True, True, True, True, True, True,    # O-
         False, True, False, True, False, True,  # O+
         False, False, True, True,               # B-
         False, False, False, True,              # B+
         True, True,                             # A-
         False, True],                           # A+
        [True, True, True,                      # O-
         False, True,                           # B-
         True]                                  # A-
    )
    if breakdowns:
        return_values = []
        for i, ci in zip(indices, compatible_indices):
            return_values.append(data.loc[:, i.flatten()].loc[:, ci])
        return tuple(return_values)

    return_values = []
    for i in indices:
        values = data.loc[:, i.flatten()].sum(axis=1)
        value = values[0]
        if print_with_stderr:
            value = f'{values[0]:.1f} ± {values[1]:.2f}'
        return_values.append(value)
    return tuple(return_values)


def abo_combos_usage_demand(totalled_combos):
    demanded = pd.DataFrame((totalled_combos.values[-1, :-1]/totalled_combos.values[-1, :-1].sum())[
                            None, :], columns=totalled_combos.columns[:-1], index=['Demand'])
    used = pd.DataFrame((totalled_combos.values[:-1, -1]/totalled_combos.values[:-1, -1].sum())[
                        None, :], columns=totalled_combos.columns[:-1], index=['Usage'])
    return pd.concat([demanded, used], axis=0)


def compile_tuning_points_all_objectives(folder, pattern='_objectives.tsv',
                                         suffix='tuning_all-vars-objs.tsv'):
    files = os.listdir(folder)
    tuning_points = [f for f in files if f.endswith('tuning_points.tsv')][0]
    files = [os.path.join(folder, x) for x in files if x.endswith(pattern)]
    files.sort(key=lambda x: os.path.getctime(os.path.join(folder, x)))

    averages = []
    for file in files:
        # Load the file into a pandas dataframe
        df = pd.read_csv(os.path.join(folder, file), sep='\t')

        # Calculate the column averages
        column_averages = df.mean()

        # Add the column averages to the list of averages
        averages.append(column_averages)

    # Concatenate the list of averages into a single dataframe
    averages_df = pd.concat(averages, axis=1)
    averages_df = averages_df.transpose()
    all_vars = pd.read_csv(os.path.join(folder,
                                        tuning_points),
                           sep='\t')
    var_names = ['immunogenicity', 'usability',
                 'substitutions', 'fifo', 'young_blood']
    all_vars = all_vars[var_names]
    vars_objs = pd.concat([all_vars, averages_df], axis=1)
    vars_objs.to_csv(os.path.join(folder,
                                  tuning_points.replace('tuning_points.tsv',
                                                        suffix)),
                     sep='\t', index=False)
    return vars_objs


def summarise_abo_mixed_allocations(abo_mixed_allocations: pd.DataFrame) -> pd.DataFrame:
    donor_target = [('O', 'B'),
                    ('O', 'A'),
                    ('O', 'AB'),
                    ('B', 'AB'),
                    ('A', 'AB')]
    data = dict()
    for combo in donor_target:
        cols = [col for col in abo_mixed_allocations.columns if (
            combo[0] + '+ ' in col or combo[0] + '- ' in col) and 'to ' + combo[1] in col]
        mean_values = [abo_mixed_allocations[col][0] for col in cols]
        std_err_values = [abo_mixed_allocations[col][1] for col in cols]
        col_vals = [sum(mean_values), np.sqrt(np.square(std_err_values).sum())]
        data.update({f'{combo[0]} to {combo[1]}': col_vals})
    return pd.DataFrame(data, index=['Mean', 'Std. Err.'])


def usability_difference_matrix():
    from BSCSimulator.util import dummy_population_phenotypes, abd_usability

    non_scd_frequencies = dummy_population_phenotypes(
        'data/bloodgroup_frequencies/ABD_old_dummy_demand.tsv')
    usability = abd_usability(
        non_scd_frequencies.frequencies.to_numpy(),
        330/3500, 1.0)
    usability_diff = usability[:, None] - usability
    compatibility = [[True] * 8,                       # O-
                     [False, True] * 4,                # O+
                     [False, False, True, True] * 2,   # B-
                     [False, False, False, True] * 2,  # B+
                     [False] * 4 + [True] * 4,         # A-
                     [False] * 4 + [False, True] * 2,  # A+
                     [False] * 6 + [True] * 2,         # AB-
                     [False] * 6 + [False, True]       # AB+
                     ]
    compatibility = np.array(compatibility)
    usability_diff[~compatibility] = np.nan
    blood_groups = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']
    usability_diff_matrix = pd.DataFrame(usability_diff,
                                         columns=[
                                             f'to {bg}' for bg in blood_groups],
                                         index=blood_groups)
    return usability_diff_matrix


def avg_stock_composition(rules, rule_files, donations=None,
                          figsize=(12, 5), dpi=200, ncol=4, bbox_to_anchor=(0.5, -0.12),
                          pdf=None):
    import matplotlib.ticker as mtick

    # rules = ['Donations', 'E0', 'E0 (No SUB)']
    # rule_files = [naive_weights, extended_no_substitution]
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
    # ax.set_title('Average Blood Stock Composition')
    ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, loc='upper center')
    # ax.grid()
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    plt.show()

# s_ax = stocks_extended[stocks_extended.columns[:8]].plot(
#     kind='line', xlabel='Days', ylabel='Units of stock in inventory', figsize=(12,8),
#     title="Stock levels of major groups for Extended rule")
# sdevs = serrs * np.sqrt(1500)
# for i in range(8):
#     s_ax.fill_between(
#         np.arange(len(serrs)), stocks_extended.iloc[:, i] + sdevs.iloc[:, i],
#         stocks_extended.iloc[:, i] - sdevs.iloc[:, i], alpha=0.25)
# s_ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

# stocks_limited = pd.read_csv(
#     os.path.join(out_folder, limited_prefix + '_stocks.tsv'), sep='\t', index_col=0)
# slerrs = pd.DataFrame(stocks_limited.iloc[:, 8:].to_numpy(), columns=stocks_limited.columns[:8])
# stocks_limited[stocks_limited.columns[:8]].plot(
#     kind='line', xlabel='Days', ylabel='Units of stock in inventory',
#     legend=True, grid=True, yerr=serrs, figsize=(12,8),
#     title="Stock levels of major groups for Limited rule")
