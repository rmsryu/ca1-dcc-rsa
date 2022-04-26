import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gp
from pandas import crosstab, DataFrame
from matplotlib.pyplot import xticks, show, subplots
from ..features.dataframe import convert_df_labels
from ..constants import enums
from IPython.core.interactiveshell import InteractiveShell


# set the parameters that control the scaling of plot elements.
sns.set_context=("talk")
# remove warnings
warnings.filterwarnings("ignore")

# Set interativeShell
InteractiveShell.ast_node_interactivity = "all"

# https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html
# Qualitative palette tab10. 
# Qualitative colormaps such tab10 will be default as rsa features are categorical. 
# If required, lightness and saturation will be adjusted for each hue or we will move toward xkcd palette if extra colors are need it.
print("Setting color pallete to tab10...")
sns.set_palette("tab10")
sns.color_palette()

def plot_chi_square_dist(X: np.array, title="", save_to_file = ""):
    fig, ax = plt.subplots(1, 1)
    
    # Fit
    df, loc, scale = stats.chi2.fit(X)

    # Display the probability density function (``pdf``):
    x = np.linspace(stats.chi2.ppf(0.01, df, loc, scale), stats.chi2.ppf(0.99, df, loc, scale), 100)
    ax.plot(x, stats.chi2.pdf(x, df, loc, scale), 'r-', lw=5, alpha=0.6, label='chi2 pdf')

    # And compare the histogram:
    plt.title(f"Chi-square")
    ax.hist(X, bins=25, density=True, alpha=0.6, color='b');
    ax.legend(loc='best', frameon=False)
    plt.savefig(save_to_file)
    plt.show()

def plot_normal_dist(X: np.array, title="", save_to_file = ""):
    
    #X = dcc_geo_df.safety_index.to_numpy();
    mu, sigma = stats.norm.fit(X);

    # Plot the histogram.
    plt.hist(X, bins=25, density=True, alpha=0.6, color='b');

    # Normal Distribution: Plot the PDF.
    xmin, xmax = plt.xlim();
    x = np.linspace(xmin, xmax, 100);
    p = stats.norm.pdf(x, mu, sigma);
    
    if title != "":
        title = "{:}\nFit Values: mu {:.2f} and sigma {:.2f}".format(title,mu, sigma);
    else:
        title = "Fit Values: mu {:.2f} and sigma {:.2f}".format(mu, sigma);

    plt.plot(x, p, 'g', linewidth=2, label="Normal Distribution")
    plt.title(title);
    if save_to_file != "":
        plt.savefig(save_to_file)
    plt.show()


def plot_crosstab_feature_analysis_by_severity(df: DataFrame, severity: bool, x :str, hue :str, stacked = False, percentages = False, save_to_file=""): 
    """ Generate

    Args:
        df (DataFrame): An RSA DataFrame
        severity (bool): Filter High / Low severity
        x (str): independent variable
        hue (str): vector or key in data Semantic variable that is mapped to determine the color of plot elements.
        stacked (bool, optional): Stacked. Defaults to False.
        percentages (bool, optional): Return percentages. Defaults to False.
    """       
    if (stacked == None):
        stacked = False
    
    fatal_df = get_crosstab(df, severity, x, hue, percentages)

    ax = fatal_df.plot(kind="bar", stacked=stacked, rot=0,figsize=(8,6))

    # print percentages on rectangle
    for c in ax.containers:
        ax.bar_label(c, label_type='center')

    # move legend up right
    ax.legend(title=hue, bbox_to_anchor=(1, 1.02), loc='upper left')
    xticks(rotation=45, ha="right");
    if(save_to_file!= ""):
        ax.figure.savefig(save_to_file)
    show()


def plot_crosstab_feature_analysis_comparison(df: DataFrame, x :str, hue :str, stacked = False, percentages = False): 
    """Generate plot for the 2 selected features

    Args:
        df (DataFrame): An RSA DataFrame
        outcome_filter (str): Filter to apply to dataframe
        x (str): independent variable
        hue (str): vector or key in data Semantic variable that is mapped to determine the color of plot elements.
        stacked (bool, optional): Stacked. Defaults to False.
        percentages (bool, optional): Return percentages. Defaults to False.
    """       
    if (stacked == None):
        stacked = False
    
    fatal_df =      get_crosstab(df, True, x, hue, percentages)
    non_fatal_df =  get_crosstab(df, False, x, hue, percentages)
    all_df =        get_crosstab(df, None, x, hue, percentages)
    
    fig, axs = subplots(1,3)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    fatal_df.plot(kind="bar", stacked=stacked, rot=0,figsize=(10,16), ax=ax1)
    non_fatal_df.plot(kind="bar", stacked=stacked, rot=0,figsize=(10,16), ax=ax2)
    all_df.plot(kind="bar", stacked=stacked, rot=0,figsize=(10,16), ax=ax3)

    # print value on rectangle
    for c in ax1.containers:
        ax1.bar_label(c, label_type='center')

    for c in ax2.containers:
        ax2.bar_label(c, label_type='center')

    for c in ax3.containers:
        ax3.bar_label(c, label_type='center')

    # legends
    ax1.legend(title=f"Fatal: {hue}")
    ax2.legend(title=f"Non-Fatal: {hue}")
    ax3.legend(title=f"Total: {hue}",)
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='x', rotation=45)
    show()

def get_crosstab(df: DataFrame, severity, x, y, percentages):
    '''
    Generate crosstab dataframe from x, y features.

    Args:
        df (DataFrame): _description_
        severity (bool): High / Low severity
        x (_type_): _description_
        y (_type_): _description_
        percentages (_type_): _description_

    Returns:
        DataFrame: Crosstab DataFrame
    '''

    if(severity != None):
        filtered_df = df[(df.severity == severity)]
    else:
        filtered_df = df

    filtered_df = filtered_df.sort_values(by=[x])
    labels_filtered_df = convert_df_labels(filtered_df)

    crosstab_df = crosstab(
        labels_filtered_df[x],
        labels_filtered_df[y],
        )
    
    if(percentages == True):
        crosstab_df.apply(lambda r: round(100*(r/r.sum()),0), axis=1)
    return crosstab_df


def plot_on_dublin_map(gdf :gp.GeoDataFrame, dcc_gdp :gp.GeoDataFrame, save_to_file = ""):
    # Set figure size
    fig, ax = plt.subplots(figsize=(30,30))
    ax.set_aspect('equal')
    dcc_gdp.plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', label='RSA Casualties', zorder=1);

    # add collision to the map
    gdf.plot(ax=ax, alpha=0.5, cmap='viridis', linewidth=0.8, legend=True, zorder=2);
            
    plt.title("RSA")
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    if save_to_file != "":
        plt.savefig(save_to_file)
    plt.show()

def plot_on_dublin_map_cluster(
    hue: str,
    cluster_centers: gp.GeoDataFrame,
    dcc_admin_gdf :gp.GeoDataFrame, 
    gdf = gp.GeoDataFrame(),
    gosafe_gdf = gp.GeoDataFrame(),
    fire_ambulance_gpf = gp.GeoDataFrame(),
    only_go_safe = None,
    highlight_go_safe = False,
    save_to_file=""):
    """Plot RSA map including, cluster centers, go safe and dublin city administrative areas

    Args:
        gdf (gp.GeoDataFrame): RSA casualties for DCC
        hue (str): Grouping variable that will produce points with different colors. Values: knn_cluster or outcome_calculated
        cluster_centers (gp.GeoDataFrame): Cluster centers, KNN
        dcc_admin_gdf (gp.GeoDataFrame): Dublin City Council Administrative areas.
        gosafe_gdf (gp.GeoDataFrame): Go Safe GeoDataFrame
        only_go_safe (_type_, optional): Plot only intersect with GoSafe. Defaults to False
        highlight_go_safe (bool, optional): Hihghligt intersect with GoSafe. Defaults to False.
        save_to_file (str, optional): Save figure to file location
    """    

    # Set figure size
    fig, ax = plt.subplots(figsize=(50,50))
    ax.set_aspect('equal')
    dcc_admin_gdf.plot(ax=ax, alpha=0.6, edgecolor='darkgrey', color='lightgrey', label='RSA Casualties', zorder=1)


    # add collision to the map
    color_mapping = {
        'Fatal': "xkcd:black", 
        'Serious': "xkcd:red",
        'Minor': "xkcd:light orange",
        'Not Injured': "xkcd:green",
        'Unknown': "xkcd:grey",
        }

    if hue == 'knn_cluster':
        color_mapping = {
            0: "tab:blue", 
            1: "tab:orange", 
            2: "tab:green", 
            3: "tab:red", 
            4: "tab:purple", 
            5: "tab:brown", 
            6: "tab:pink", 
            7: "tab:gray", 
            8: "tab:olive", 
            9: "tab:cyan",
            10: "xkcd:dark blue",
            11: "xkcd:dark orange",
            12: "xkcd:dark green",
            13: "xkcd:dark red",
            14: "xkcd:dark purple",
        }
    if len(gdf.columns) > 0:
        # Add gosafe background layer
        if gosafe_gdf is not None:
            gosafe_gdf.plot(ax=ax)
            if 'is_gosafe' in gdf.columns:
                if highlight_go_safe:
                    gdf[gdf.is_gosafe == True].plot(ax=ax, linewidth=30, legend=True, zorder=2, marker='o', color='cyan', markersize=30)

        # Add rsa casualties layer
        gdf_plot = gp.GeoDataFrame();
        if only_go_safe != None:
            gdf_plot = gdf[gdf['is_gosafe']==only_go_safe]
        else:
            gdf_plot = gdf
        
        # Loop through each attribute type and plot it using the colors assigned in the dictionary
        for ctype, data in gdf_plot.groupby(hue):
            # Define the color for each group using the dictionary
            color = color_mapping[ctype]
            # Plot each group using the color defined above
            data.plot(
                    ax=ax,
                    color=color,
                    column=hue,
                    categorical=True,
                    label=ctype,
                    legend=True,
                    legend_kwds={'loc':'upper left','fontsize':25,'frameon':False},
                    zorder=2)

    # Add culster centers
    cluster_centers.plot(ax=ax, linewidth=10, legend=True, marker='^', color='#f00', markersize=150, zorder=3)
    
    for x, y, label in zip(cluster_centers.geometry.x, cluster_centers.geometry.y, cluster_centers.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", size=30)

    if len(fire_ambulance_gpf.columns) > 0:
        fire_ambulance_gpf.plot(ax=ax, linewidth=10, legend=True, marker='x', color='#0f0', markersize=150, zorder=4)
    
        for x, y, label in zip(fire_ambulance_gpf.geometry.x, fire_ambulance_gpf.geometry.y, fire_ambulance_gpf.name):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", size=30)
    
    
    # Customize fonts
    ax.legend(prop={'size': 25})
    plt.title('RSA', fontdict={'fontsize':30})
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    if save_to_file != "":
        plt.savefig(save_to_file)
    plt.show()