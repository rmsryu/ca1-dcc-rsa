# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause
# REF: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#compare-the-effect-of-different-scalers-on-data-with-outliers

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from ..constants import enums

class ScalerComparison:
    
    def __init__(self, rsa_df :pd.DataFrame):
        # scale the output between 0 and 1 for the colorbar
        self.features = [enums.feature_names.LATITUDE, enums.feature_names.RADIUS_500M]
        self.feature_mapping = {
            enums.feature_names.RADIUS_500M: "Radious 500m Collision",
            enums.feature_names.AGE: "Group age",
            enums.feature_names.VEHICLE_TYPE: "vehicle_type",
            enums.feature_names.LATITUDE: "latitude",
            enums.feature_names.LONGITUDE: "longitude",
        }

        self.X_full = rsa_df
        self.y_full = rsa_df.outcome_calculated
        self.X = self.X_full[self.features]
        self.y = minmax_scale(self.y_full)

        self.distributions = [
            ("Unscaled data", self.X),
            ("Data after standard scaling", StandardScaler().fit_transform(self.X)),
            ("Data after min-max scaling", MinMaxScaler().fit_transform(self.X)),
            ("Data after max-abs scaling", MaxAbsScaler().fit_transform(self.X)),
            ("Data after robust scaling", RobustScaler(quantile_range=(25, 75)).fit_transform(self.X)),
            ("Data after power transformation (Yeo-Johnson)", PowerTransformer(method="yeo-johnson").fit_transform(self.X)),
            ("Data after power transformation (Box-Cox)",PowerTransformer(method="box-cox").fit_transform(self.X)),
            ("Data after quantile transformation (uniform pdf)",QuantileTransformer(output_distribution="uniform").fit_transform(self.X)),
            ("Data after quantile transformation (gaussian pdf)",QuantileTransformer(output_distribution="normal").fit_transform(self.X)),
            ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(self.X))
        ]
        
        # plasma does not exist in matplotlib < 1.5
        self.cmap = getattr(cm, "plasma_r", cm.hot_r)


    def create_axes(self, title, figsize=(16, 6)):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)

        # define the axis for the first plot
        left, width = 0.1, 0.22
        bottom, height = 0.1, 0.7
        bottom_h = height + 0.15
        left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, 0.05, height]

        ax_scatter = plt.axes(rect_scatter)
        ax_histx = plt.axes(rect_histx)
        ax_histy = plt.axes(rect_histy)

        # define the axis for the zoomed-in plot
        left = width + left + 0.2
        left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, 0.05, height]

        ax_scatter_zoom = plt.axes(rect_scatter)
        ax_histx_zoom = plt.axes(rect_histx)
        ax_histy_zoom = plt.axes(rect_histy)

        # define the axis for the colorbar
        left, width = width + left + 0.13, 0.01

        rect_colorbar = [left, bottom, width, height]
        ax_colorbar = plt.axes(rect_colorbar)

        return (
            (ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar,
        )


    def plot_distribution(self, axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
        ax, hist_X1, hist_X0 = axes

        ax.set_title(title)
        ax.set_xlabel(x0_label)
        ax.set_ylabel(x1_label)

        # The scatter plot
        colors = self.cmap(y)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

        # Removing the top and the right spine for aesthetics
        # make nice axis layout
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))

        # Histogram for axis X1 (feature 5)
        hist_X1.set_ylim(ax.get_ylim())
        hist_X1.hist(
            X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
        )
        hist_X1.axis("off")

        # Histogram for axis X0 (feature 0)
        hist_X0.set_xlim(ax.get_xlim())
        hist_X0.hist(
            X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
        )
        hist_X0.axis("off")

    def make_plot(self, item_idx):
        title, X = self.distributions[item_idx]
        ax_zoom_out, ax_zoom_in, ax_colorbar = self.create_axes(title)
        axarr = (ax_zoom_out, ax_zoom_in)
        self.plot_distribution(
            axarr[0],
            X,
            self.y,
            hist_nbins=200,
            x0_label=self.feature_mapping[self.features[0]],
            x1_label=self.feature_mapping[self.features[1]],
            title="Full data",
        )

        # zoom-in
        zoom_in_percentile_range = (0, 99)
        cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
        cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

        non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
            X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
        )
        self.plot_distribution(
            axarr[1],
            X[non_outliers_mask],
            self.y[non_outliers_mask],
            hist_nbins=50,
            x0_label=self.feature_mapping[self.features[0]],
            x1_label=self.feature_mapping[self.features[1]],
            title="Zoom-in",
        )
        norm = mpl.colors.Normalize(self.y_full.min(), self.y_full.max())
        mpl.colorbar.ColorbarBase(
            ax_colorbar,
            cmap=self.cmap,
            norm=norm,
            orientation="vertical",
            label="Color mapping for values of y",
        )

        #Plot normalized
        # scaler_df  =   pd.DataFrame(X.tolist(), columns=['scaled']) 
        # scaler_df['outcome_calculated'] = self.X_full['outcome_calculated']
        # sns.histplot(data=scaler_df,bins=50, x="scaled", hue="outcome_calculated", kde=True)