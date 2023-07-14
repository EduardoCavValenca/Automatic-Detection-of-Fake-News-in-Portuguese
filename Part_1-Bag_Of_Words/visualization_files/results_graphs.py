import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap
sns.set_theme(style="whitegrid", palette="Set2", color_codes=True)

def graph_compare_feature_num_accuracys(df_plot: pd.DataFrame, title: str = None, ax: plt.Axes = None):
    df_plot = df_plot.reset_index()
    df_plot = df_plot.melt(id_vars=["vectorizer"], var_name="metric", value_name="accuracys")
    df_plot["vectorizer"] = df_plot["vectorizer"].replace({"256":"256 features", "1024":"1024 features", "4096":"4096 features", "max":"max features"})

    sns.set_theme(style="whitegrid", palette="Set2", color_codes=True)

    if not ax:
    # Initialize the matplotlib figure
        f, ax = plt.subplots(figsize=(6, 4))

    # Plot the total crashes
    sns.barplot(x="vectorizer", y="accuracys", hue="metric", data=df_plot, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='%1.2f', color='black', size=8)

    # Add a legend and informative axis label
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3)

    ax.set_ylim(0, 1)
    ax.set_title(title)
    return ax

def graph_compare_feature_num_train(df_plot: pd.DataFrame, title: str = None, ax: plt.Axes = None):
    df_plot = df_plot.reset_index()
    df_plot = df_plot.melt(id_vars=["vectorizer"], var_name="metric", value_name="seconds")
   
    sns.set_theme(style="whitegrid", palette="Set2", color_codes=True)

    if not ax:
    # Initialize the matplotlib figure
        f, ax = plt.subplots(figsize=(6, 4))

    #cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 

    # Plot the total crashes
    sns.barplot(x="seconds", y="vectorizer", data=df_plot, ax=ax, palette=colors_from_values(df_plot["seconds"], "OrRd"))

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='%1.2f', color='black', size=8)

    ax.set_title(title)
    plt.ylabel("Number of words")
    plt.xlabel("Time for prediction (s)")
    return ax

def graph_compare_datasets_accuracy(df_plot: pd.DataFrame, title: str = None):
    df_plot = df_plot.reset_index()
    df_plot = df_plot.melt(id_vars=["dataset"], var_name="metric", value_name="accuracys")

    sns.set_theme(style="whitegrid", palette="Set2", color_codes=True)

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(4, 4))

    # Plot the total crashes
    sns.barplot(x="dataset", y="accuracys", hue="metric", data=df_plot, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='%1.2f', color='black', size=8)

    # Add a legend and informative axis label
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3)

    ax.set_ylim(0, 1)
    ax.set_title(title)
    return ax


def graph_compare_face_check_accuracy(df_plot: pd.DataFrame, title: str = None):
    df_plot = df_plot.reset_index()
    df_plot = df_plot.melt(id_vars=["algorithm"], var_name="metric", value_name="accuracys")

    sns.set_theme(style="whitegrid", palette="Set2", color_codes=True)

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(4, 4))

    # Plot the total crashes
    sns.barplot(x="algorithm", y="accuracys", hue="metric", data=df_plot, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='%1.2f', color='black', size=8)

    # Add a legend and informative axis label
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3)

    ax.set_ylim(0, 1)
    ax.set_title(title)
    return ax

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return palette