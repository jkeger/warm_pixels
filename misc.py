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
    "text.usetex": True,
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


# ========
# Plotting functions
# ========
def set_std_form_axes(ax, force_x=False, force_y=False):
    """ Set the axis labels to use standard form, if the current limits are very
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


def set_std_form_cbar(cbar, A1_colour, force=False):
    """ Set a colour bar's axis labels to use standard form, if the current
        limits are very large or small.

        Args:
            cbar (plt colorbar)
                The colour bar object.

            A1_colour ([float,])
                Array of floats that are setting the colour values.

            force (opt. bool)
                False:      (Default) Only set stardard form if the axis limits
                            are very large or small.
                True:       Set standard form regardless of the current limits.
    """
    # Boundary limits
    lim_min = 1e-2
    lim_max = 1e3

    # Current axis limits
    c_min = abs(np.amin(A1_colour))
    c_max = abs(np.amax(A1_colour))

    # Set to standard form
    if (c_min < lim_min and c_min != 0) or lim_max < c_max or force:
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()


def set_large_ticks(ax, do_x=True, do_y=True):
    """ Set larger ticks on plot axes, especially for logarithmic scales.

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


def set_nice_limits(
    ax,
    A1_x,
    A1_y,
    frac_x=0.05,
    frac_y=0.05,
    set_x=True,
    set_y=True,
    min_x=None,
    min_y=None,
    max_x=None,
    max_y=None,
):
    """ Set the axes limits to a nice small distance away from the highest and
        lowest points. Also set suitable logarithmic axis labels.

        Args:
            ax (plt axes)
                The plot axes.

            A1_x, A1_y ([float])
                The data point x and y values.

            frac_x, frac_y (float)
                The fraction of the data's range by which to extend the axes.
                Default 1/20.

            set_x, set_y (opt. bool)
                True:       (Default) Set new limits for that axis.
                False:      Do not set new limits for that axis.

            min_x, min_y (opt. bool)
                If not None then force the minimum value for that axis.

            max_x, max_y (opt. float)
                If not None then force the maximum value for that axis.

        If a logarithmic axis covers less than two decades, add standard form
        axis labels for some minor ticks as well.
    """
    # ========
    # Check args
    # ========
    set_x = check_bool(set_x)
    set_y = check_bool(set_y)
    min_x = check_none(min_x)
    min_y = check_none(min_y)
    max_x = check_none(max_x)
    max_y = check_none(max_y)
    A1_x = np.array(A1_x)
    A1_y = np.array(A1_y)

    # Check if either axis is logarithmic
    log_x = False
    log_y = False
    if ax.get_xscale() == "log":
        log_x = True
    if ax.get_yscale() == "log":
        log_y = True

    # ========
    # Axis limits
    # ========
    x_min = np.nanmin(A1_x)
    x_max = np.nanmax(A1_x)
    y_min = np.nanmin(A1_y)
    y_max = np.nanmax(A1_y)
    if log_x:
        # Use the lowest positive value for the log scale minimum
        if x_min <= 0:
            x_min = np.nanmin(A1_x[A1_x > 0])

        x_min = np.log10(x_min)
        x_max = np.log10(x_max)
    if log_y:
        # Use the lowest positive value for the log scale minimum
        if y_min <= 0:
            y_min = np.nanmin(A1_y[A1_y > 0])

        y_min = np.log10(y_min)
        y_max = np.log10(y_max)
    if min_x is not None:
        x_min = min_x
    if min_y is not None:
        y_min = min_y
    if max_x is not None:
        x_max = max_x
    if max_y is not None:
        y_max = max_y

    # Set the limits to be a fraction of the total range outside the points
    x_1 = x_min - (x_max - x_min) * frac_x
    x_2 = x_max + (x_max - x_min) * frac_x
    y_1 = y_min - (y_max - y_min) * frac_y
    y_2 = y_max + (y_max - y_min) * frac_y
    if log_x:
        x_1 = 10 ** (x_min - (x_max - x_min) * frac_x)
        x_2 = 10 ** (x_max + (x_max - x_min) * frac_x)
    if log_y:
        y_1 = 10 ** (y_min - (y_max - y_min) * frac_y)
        y_2 = 10 ** (y_max + (y_max - y_min) * frac_y)

    if min_x is not None:
        x_1 = min_x
    if min_y is not None:
        y_1 = min_y
    if max_x is not None:
        x_2 = max_x
    if max_y is not None:
        y_2 = max_y

    if set_x:
        ax.set_xlim(x_1, x_2)
    if set_y:
        ax.set_ylim(y_1, y_2)

    # ========
    # Logarithmic labels
    # ========
    if log_x and set_x:
        if np.log10(x_2) - np.log10(x_1) < 2:
            # Visible ticks
            dec = 10 ** (np.floor(np.log10(x_1)))
            A1_x_tick = (
                np.append(
                    np.arange(np.floor(x_1 / dec) + 1, 10, 1),
                    np.arange(1, np.floor(x_2 / (dec * 10)), 1) * 10,
                )
            ) * dec

            # Tick first digits and order of magnitude
            A1_x_tick_digit = np.array([int(str(tick)[0]) for tick in A1_x_tick])
            A1_x_tick_dec = np.log10(A1_x_tick / A1_x_tick_digit)

            try:
                # Standard form labels
                A1_x_ticklabel = np.array(
                    [
                        r"%d$\times$10$^%d$"
                        % (A1_x_tick_digit[i_tick], A1_x_tick_dec[i_tick])
                        for i_tick in range(len(A1_x_tick))
                    ]
                )

                # No labels other than 5 or 3 and 5
                if np.log10(x_2) - np.log10(x_1) < 1:
                    sel_ticks = np.where(
                        (A1_x_tick / (10 ** np.floor(np.log10(A1_x_tick))) != 3)
                        & (A1_x_tick / (10 ** np.floor(np.log10(A1_x_tick))) != 5)
                    )[0]
                else:
                    sel_ticks = np.where(
                        A1_x_tick / (10 ** np.floor(np.log10(A1_x_tick))) != 5
                    )[0]

                A1_x_ticklabel[sel_ticks] = ""

                ax.xaxis.set_ticks(A1_x_tick, minor=True)
                ax.xaxis.set_ticklabels(A1_x_ticklabel, minor=True)
            except OverflowError:
                ###get from x_tick_digit = 0, but haven't thought why
                pass
    if log_y and set_y:
        if np.log10(y_2) - np.log10(y_1) < 2:
            # Visible ticks
            dec = 10 ** (np.floor(np.log10(y_1)))
            A1_y_tick = (
                np.append(
                    np.arange(np.floor(y_1 / dec) + 1, 10, 1),
                    np.arange(1, np.floor(y_2 / (dec * 10)), 1) * 10,
                )
            ) * dec

            # Tick first digits and order of magnitude
            A1_y_tick_digit = np.array([int(str(tick)[0]) for tick in A1_y_tick])
            A1_y_tick_dec = np.log10(A1_y_tick / A1_y_tick_digit)

            # Standard form labels
            try:
                A1_y_ticklabel = np.array(
                    [
                        r"%d$\times$10$^%d$"
                        % (A1_y_tick_digit[i_tick], A1_y_tick_dec[i_tick])
                        for i_tick in range(len(A1_y_tick))
                    ]
                )

                if np.log10(y_2) - np.log10(y_1) < 1:
                    sel_ticks = np.where(
                        (A1_y_tick / (10 ** np.floor(np.log10(A1_y_tick))) != 3)
                        & (A1_y_tick / (10 ** np.floor(np.log10(A1_y_tick))) != 5)
                    )[0]
                else:
                    sel_ticks = np.where(
                        A1_y_tick / (10 ** np.floor(np.log10(A1_y_tick))) != 5
                    )[0]

                A1_y_ticklabel[sel_ticks] = ""

                ax.yaxis.set_ticks(A1_y_tick, minor=True)
                ax.yaxis.set_ticklabels(A1_y_ticklabel, minor=True)
            except OverflowError:
                pass


def set_font_size(ax, cbar=None, fontsize=22):
    """ Set the font size for all plot text.

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


def set_nice_ticks(ax, x_or_y, x_max, x_min=0):
    """ Set nice tick spacing for the x- or y-axis limits.

        Args:
            ax (plt axes)
                The plot axes.

            x_or_y (str)
                "x"     Set the x ticks.
                "y"     Set the y ticks.

            x_max (float)
                The data maximum.

            x_min (opt. float)
                The data minimum. Default 0.
    """
    ### Temperamental! Sometimes works but not tested thoroughly enough...

    check_option(x_or_y, ["x", "y"])

    # Set spacing to be 1 order of mag less than the axis range
    tick_space = 10 ** int(int(np.log10(x_max - x_min)) - 1)
    num_ticks = (x_max - x_min) / tick_space

    # Smaller spacing if too few ticks
    if num_ticks < 2:
        ax.xaxis.set_minor_locator(MultipleLocator(tick_space * 0.25))
        tick_space /= 2

    # Larger spacing with minor ticks if too many major ticks
    elif num_ticks > 10:
        ax.xaxis.set_minor_locator(MultipleLocator(tick_space * 2.5))
        tick_space *= 5
    elif num_ticks > 5:
        ax.xaxis.set_minor_locator(MultipleLocator(tick_space))
        tick_space *= 2

    # Set a nice first tick value
    x_min = tick_space * (x_min // tick_space)

    A1_ticks = np.arange(x_min, x_max * 1.05, tick_space)

    # Set the ticks
    if x_or_y == "x":
        ax.set_xticks(A1_ticks)
    else:
        ax.set_yticks(A1_ticks)


def nice_plot(ax=None, cbar=None, fontsize=22):
    """ Call my standard nice-plot-making functions, default values only. """
    if ax is None:
        ax = plt.gca()
    set_large_ticks(ax)
    set_std_form_axes(ax)
    set_font_size(ax, cbar=cbar, fontsize=fontsize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()


def plot_hist(ax, A1_hist, A1_bin_edge, c="k", ls="-", lw=1.7, alpha=1, label=None):
    """ Plot a nice histogram with no vertical lines in the middle.

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
