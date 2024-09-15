import numpy as np
import matplotlib as mpl
import os

# Use a non-interactive backend if no display found
if os.environ.get("DISPLAY", "") == "" or os.environ.get("NONINTERACTIVE") is not None:
    mpl.use("Agg")
    # Avoid weird latex plotting error
    preamble = r"\usepackage{gensymb}" r"\newcommand{\mathdefault}[1][]{}"
else:
    preamble = r"\usepackage{gensymb}"
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings


# ========
# Matplotlib defaults
# ========
font_size = 20
params = {
    "backend": "ps",
    "text.latex.preamble": preamble,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "font.size": font_size,
    "legend.fontsize": font_size - 4,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "text.usetex": False,
    "figure.figsize": [9, 9],
    "font.family": "serif",
    "savefig.dpi": 100,
    "legend.framealpha": 0.85,
    "lines.linewidth": 1.7,
}
mpl.rcParams.update(params)

# List of nice, contrasting colours for general use
# The first few are well-separated in black and white too
A1_c = [
    "#1199ff",
    "#ee4400",
    "#7711dd",
    "#44dd44",
    "#ffdd00",
    "#55eedd",
    "#ff66bb",
    "#775533",
    "#707070",
    "#ccaaee",
    "#0000cc",
    "#cc0000",
    "#660077",
    "#007700",
    "#ff9922",
    "#aaccff",
    "#ffcccc",
    "#cc8877",
    "#111111",
    "#aaaaaa",
]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=A1_c)

# Line styles
ls_dot = (0, (1, 3))
ls_dash = (0, (4, 3))


# ========
# Plotting functions
# ========
def set_std_form_axes(ax, force_x=False, force_y=False):
    """Set the axis labels to use standard form, if the current limits are very
    large or small (and the scale is linear).

    Args:
        ax (plt axes)
            The plot axes.

        force_x, force_y (opt. bool)
            False:      (Default) Only set stardard form if the axis limits
                        are very large or small.
            True:       Set standard form regardless of the current limits.
    """
    # Check if either axis is logarithmic
    if ax.get_xscale() == "log":
        log_x = True
    else:
        log_x = False
    if ax.get_yscale() == "log":
        log_y = True
    else:
        log_y = False

    # Boundary labels
    lim_min = 1e-3
    lim_max = 1e4

    # ========
    # Current smallest and largest axis labels
    # ========
    A1_x_tick = abs(ax.get_xticks())
    A1_y_tick = abs(ax.get_yticks())
    # Ignore 0
    A1_x_tick = A1_x_tick[A1_x_tick != 0]
    A1_y_tick = A1_y_tick[A1_y_tick != 0]

    x_min = np.amin(A1_x_tick)
    x_max = np.amax(A1_x_tick)
    y_min = np.amin(A1_y_tick)
    y_max = np.amax(A1_y_tick)

    # ========
    # Set to standard form
    # ========
    # x axis
    if (x_min < lim_min or lim_max < x_max or force_x) and not log_x:
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        ax.xaxis.set_major_formatter(formatter)
        try:
            ax.get_xaxis().get_major_formatter().set_powerlimits((0, 0))
        except AttributeError:
            pass
    # y axis
    if (y_min < lim_min or lim_max < y_max or force_y) and not log_y:
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        ax.yaxis.set_major_formatter(formatter)
        try:
            ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
        except AttributeError:
            pass


def set_large_ticks(ax, do_x=True, do_y=True):
    """Set larger ticks on plot axes, especially for logarithmic scales.

    Args:
        ax (plt axes)
            The plot axes.
    """
    # Check if either axis is logarithmic
    log_x = False
    log_y = False
    if ax.get_xscale() == "log":
        log_x = True
    if ax.get_yscale() == "log":
        log_y = True

    # Tick sizes
    width_log = 1.0
    width_lin = 0.8
    width_min = 0.8
    length_log = 10
    length_lin = 6
    length_min = 4

    # First set all major and minor ticks to the logarithmic sizes
    if do_x:
        ax.xaxis.set_tick_params(width=width_log, length=length_log, which="major")
        ax.xaxis.set_tick_params(width=width_log, length=length_lin, which="minor")
    if do_y:
        ax.yaxis.set_tick_params(width=width_log, length=length_log, which="major")
        ax.yaxis.set_tick_params(width=width_log, length=length_lin, which="minor")

    # Reset linear ticks ((this weird order seems to work best))
    if do_x and not log_x:
        ax.xaxis.set_tick_params(width=width_lin, length=length_lin, which="major")
        ax.xaxis.set_tick_params(width=width_min, length=length_min, which="minor")
    if do_y and not log_y:
        ax.yaxis.set_tick_params(width=width_lin, length=length_lin, which="major")
        ax.yaxis.set_tick_params(width=width_min, length=length_min, which="minor")


def set_font_size(ax, cbar=None, fontsize=22):
    """Set the font size for all plot text.

    Args:
        ax (plt axes)
            The plot axes.

        cbar (opt. plt colour bar)
            A colour bar object.
            None:       (Default)

        fontsize (opt. float)
            The font size.
            20:         (Default)
    """
    # Main labels
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)
    # Minor tick labels
    for item in [] + ax.xaxis.get_minorticklabels() + ax.yaxis.get_minorticklabels():
        item.set_fontsize(fontsize - 2)
    # Colour bar
    if cbar:
        cbar.ax.tick_params(labelsize=fontsize - 2)
        cbar.ax.yaxis.label.set_font_properties(
            mpl.font_manager.FontProperties(size=fontsize)
        )


def nice_plot(ax=None, cbar=None, fontsize=22):
    """Call my standard nice-plot-making functions, default values only."""
    if ax is None:
        ax = plt.gca()
    set_large_ticks(ax)
    set_std_form_axes(ax)
    set_font_size(ax, cbar=cbar, fontsize=fontsize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()


def plot_hist(ax, A1_hist, A1_bin_edge, c="k", ls="-", lw=1.7, alpha=1, label=None):
    """Plot a nice histogram with no vertical lines in the middle.

    Parameters
    ----------
    ax : plt axes
        The plot axes.

    A1_hist, A1_bin_edge : [float]
        The histogram and bin arrays returned by numpy.histogram(): a list of
        number counts or density values in each bin and a list of bin edges
        including both outside values.

    c, ls, lw : str (opt.)
        The colour, linestyle, and linewidth. Default solid black line.

    alpha : float (opt.)
        The opacity. Default solid.

    label : str (opt.)
        The label, if required.
    """
    # Append values to complete the plot with vertical lines at both ends
    A1_hist = np.append(-np.inf, A1_hist)
    A1_hist = np.append(A1_hist, -np.inf)
    A1_bin_edge = np.append(A1_bin_edge, A1_bin_edge[-1])

    ax.plot(
        A1_bin_edge,
        A1_hist,
        c=c,
        ls=ls,
        lw=lw,
        drawstyle="steps",
        alpha=alpha,
        label=label,
    )
