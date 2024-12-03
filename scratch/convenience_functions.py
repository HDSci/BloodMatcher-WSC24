import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick


def objectives_stats(file_path):
    output_df = pd.read_csv(file_path, sep='\t')
    columns = list(output_df.columns)
    return_value = []
    for column in columns:
        return_value.append((output_df[column].mean(),
                             output_df[column].std() / np.sqrt(len(output_df))))
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


def avg_stock_composition(rules, rule_files, donations=None,
                          figsize=(12, 5), dpi=200, ncol=4, bbox_to_anchor=(0.5, -0.12),
                          pdf=None):
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
