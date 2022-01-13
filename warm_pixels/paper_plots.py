"""Paper and test plots etc in addition to hst_warm_pixels.py

Parameters
----------
run : [str]
    Which function(s) to run, chosen by the function name or a substring of the
    name. Accepts multiple values. Defaults to run all.

--pdf, -p
    Save as pdf not png.
"""

import argparse

import matplotlib.patheffects as path_effects
from matplotlib.patches import ConnectionPatch

import arcticpy as cti
from warm_pixels.hst_warm_pixels import *
from warm_pixels.misc import *
from warm_pixels.pixel_lines import PixelLineCollection


#
# RJM: Jacob set up variables here, thinking they had global scope. They don't.
#
# Example dataset and image   
# dataset = Dataset("07_2020")  # 2020/07/31, day 6727, 8 images
# image_name = "jdrwc3fcq_raw"
# image_path = dataset.path + image_name + ".fits"
# quadrant = "D"
# n_iterations = 1
# cor_path = dataset.path + image_name + "_cor_iter%d.fits" % n_iterations


# ========
# Utilities
# ========
def prep_parser():
    """Prepare the sys args parser."""
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument(
        "run", nargs="*", default=["all"], type=str, help="Which function(s) to run."
    )

    # Optional arguments
    parser.add_argument(
        "-p",
        "--pdf",
        action="store_true",
        default=False,
        required=False,
        help="Save as pdf not png.",
    )

    return parser


def run(name):
    """Whether to run the function with this name."""
    # Run if either running all or the name (or part of it) was provided
    return args.run == ["all"] or any([run in name for run in args.run])


def save_fig(Fp_save, do_pdf=False):
    """Save a figure and print the file path"""
    Fp_path = "paper_plots/"
    if not os.path.exists(Fp_path):
        os.mkdir(Fp_path)
    if do_pdf:
        Fp_save = Fp_path + Fp_save + ".pdf"
    else:
        Fp_save = Fp_path + Fp_save + ".png"
    plt.savefig(Fp_save, dpi=200)
    print("Saved %s" % Fp_save[-64:])


# ========
# Functions
# ========
def example_image_zooms(image_path, cor_path, quadrant, do_pdf=False, use_corrected=False):
    """Example HST ACS image with CTI trails"""

    if use_corrected:
        file_path = cor_path
    else:
        file_path = image_path

    # Load the image
    image = aa.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native

    print("image read")

    # Figure
    fig = plt.figure(figsize=(18, 9), constrained_layout=False)
    widths = [1.3, 1, 0.13, 0.06]
    heights = [1, 1]
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 3])
    gs.update(wspace=0.01, hspace=0)

    # Zoom regions
    n_row, n_col = image.shape
    col_0 = n_col - 345
    col_1 = col_0 + 300
    row1_0 = 795
    row1_1 = row1_0 + 180
    row2_0 = n_row - 235
    row2_1 = row2_0 + 180

    # Plot the image and zooms
    vmin, vmax = 0, 400
    im1 = ax1.imshow(X=image, aspect="equal", vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(
        X=image[row1_0:row1_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row1_1, row1_0],
    )
    im3 = ax3.imshow(
        X=image[row2_0:row2_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row2_1, row2_0],
    )

    # Zoom lines
    c_zm = "0.7"
    lw_zm = 1.4
    ax1.plot(
        [col_0, col_1, col_1, col_0, col_0],
        [row1_0, row1_0, row1_1, row1_1, row1_0],
        c=c_zm,
        lw=lw_zm,
    )
    ax1.plot(
        [col_0, col_1, col_1, col_0, col_0],
        [row2_0, row2_0, row2_1, row2_1, row2_0],
        c=c_zm,
        lw=lw_zm,
    )
    for xyA, xyB, axB in [
        [(col_0, row1_0), (0, 1), ax2],
        [(col_0, row1_1), (0, 0), ax2],
        [(col_0, row2_0), (0, 1), ax3],
        [(col_0, row2_1), (0, 0), ax3],
    ]:
        ax1.add_artist(
            ConnectionPatch(
                xyA=xyA,
                xyB=xyB,
                coordsA=ax1.transData,
                coordsB=axB.transAxes,
                color=c_zm,
                lw=lw_zm,
            )
        )

    # Axes etc
    cbar = plt.colorbar(im1, cax=cax, extend="max")
    cbar.set_label(r"Flux (e$^-$)")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.tick_right()
    ax3.yaxis.tick_right()
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    for ax in [ax1.xaxis, ax1.yaxis]:
        ax.set_major_locator(mpl.ticker.MultipleLocator(base=500))
        ax.set_minor_locator(mpl.ticker.MultipleLocator(base=100))
    for ax in [ax2.xaxis, ax2.yaxis, ax3.xaxis, ax3.yaxis]:
        ax.set_major_locator(mpl.ticker.MultipleLocator(base=50))
        ax.set_minor_locator(mpl.ticker.MultipleLocator(base=10))
    cbar.ax.minorticks_on()
    cbar.ax.yaxis.set_ticks(np.arange(vmin, vmax + 1, 100))
    cbar.ax.yaxis.set_ticks(np.arange(50, vmax + 1, 100), minor=True)
    for ax in [ax1, ax2, ax3, cax]:
        set_large_ticks(ax)

    # Save
    if use_corrected:
        save_fig("example_image_corrected", do_pdf)
    else:
        save_fig("example_image_zooms", do_pdf)


def example_image_corrected(image_path, cor_path, quadrant, n_iterations, do_pdf=False):
    """Example HST ACS image with CTI trails removed by arctic"""

    # Remove CTI
    if True:
        # Load each quadrant of the image
        image_A, image_B, image_C, image_D = [
            aa.acs.ImageACS.from_fits(
                file_path=image_path,
                quadrant_letter=quadrant_letter,
                bias_subtract_via_bias_file=True,
                bias_subtract_via_prescan=True,
            ).native
            for quadrant_letter in ["A", "B", "C", "D"]
        ]

        # CTI model
        traps = [
            cti.TrapInstantCapture(density=0.60, release_timescale=0.74),
            cti.TrapInstantCapture(density=1.60, release_timescale=7.70),
            cti.TrapInstantCapture(density=1.35, release_timescale=37.0),
        ]
        roe = cti.ROE()
        ccd = cti.CCD(full_well_depth=84700, well_fill_power=0.478)

        print(traps)

        # Remove CTI
        def remove_cti(image):

            print("fff")
            print(traps)
            return cti.remove_cti(
                image=image,
                n_iterations=n_iterations,
                parallel_roe=roe,
                parallel_ccd=ccd,
                parallel_traps=traps,
                parallel_express=10,
            )

        print("hw2")

        image_out_A = remove_cti(image_A)
        print("hw3")
        image_out_B, image_out_C, image_out_D = [
            remove_cti(image) for image in [image_B, image_C, image_D]
        ]

        # Save the corrected image
        aa.acs.output_quadrants_to_fits(
            file_path=cor_path,
            quadrant_a=image_out_A,
            quadrant_b=image_out_B,
            quadrant_c=image_out_C,
            quadrant_d=image_out_D,
            header_a=image_A.header,
            header_b=image_B.header,
            header_c=image_C.header,
            header_d=image_D.header,
            overwrite=True,
        )

        print("Saved %s" % cor_path[-36:])

    # Load the images
    image = aa.acs.ImageACS.from_fits(
        file_path=image_path,
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native
    image_cor = aa.acs.ImageACS.from_fits(
        file_path=cor_path,
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native
    # Diff
    image_diff = image_cor - image

    # Figure
    fig = plt.figure(figsize=(17.4, 9), constrained_layout=False)
    widths = [1, 1, 0.03, 0.06, 0.17, 0.06]
    heights = [1, 1]
    gs = fig.add_gridspec(nrows=2, ncols=6, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    cax1 = fig.add_subplot(gs[:, 3])
    cax2 = fig.add_subplot(gs[:, 5])
    gs.update(wspace=0, hspace=0.03)

    # Zoom regions
    n_row, n_col = image.shape
    col_0 = n_col - 345
    col_1 = col_0 + 300
    row1_0 = 795
    row1_1 = row1_0 + 180
    row2_0 = n_row - 235
    row2_1 = row2_0 + 180

    # Plot the image and zooms
    vmin, vmax = 0, 400
    im1 = ax1.imshow(
        X=image_cor[row1_0:row1_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row1_1, row1_0],
    )
    im2 = ax2.imshow(
        X=image_cor[row2_0:row2_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row2_1, row2_0],
    )
    diffmin, diffmax = -60, 60
    im3 = ax3.imshow(
        X=image_diff[row1_0:row1_1, col_0:col_1],
        aspect="equal",
        cmap=plt.cm.binary_r,
        vmin=diffmin,
        vmax=diffmax,
        extent=[col_0, col_1, row1_1, row1_0],
    )
    im4 = ax4.imshow(
        X=image_diff[row2_0:row2_1, col_0:col_1],
        aspect="equal",
        cmap=plt.cm.binary_r,
        vmin=diffmin,
        vmax=diffmax,
        extent=[col_0, col_1, row2_1, row2_0],
    )

    # Axes etc
    cbar1 = plt.colorbar(im1, cax=cax1, extend="max")
    cbar1.set_label(r"Flux (e$^-$)")
    cbar2 = plt.colorbar(im3, cax=cax2, extend="both")
    cbar2.set_label(r"Corrected $-$ Original (e$^-$)")
    ax1.xaxis.set_visible(False)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax1.set_ylabel("Row")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax4.set_xlabel("Column")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=50))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=10))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=50))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=10))
    cbar1.ax.minorticks_on()
    cbar1.ax.yaxis.set_ticks(np.arange(vmin, vmax + 1, 100))
    cbar1.ax.yaxis.set_ticks(np.arange(50, vmax + 1, 100), minor=True)
    cbar2.ax.minorticks_on()
    cbar2.ax.yaxis.set_ticks(np.arange(diffmin, diffmax + 1, 20))
    cbar2.ax.yaxis.set_ticks(np.arange(diffmin + 10, diffmax + 1, 20), minor=True)
    for ax in [ax1, ax2, ax3, ax4, cax1, cax2]:
        set_large_ticks(ax)

    # Save
    save_fig("example_image_corrected", do_pdf)


def found_warm_pixels(image_path, quadrant, do_pdf=False):
    """Example HST ACS image with identified warm pixels"""

    # Load the image
    image = aa.acs.ImageACS.from_fits(
        file_path=image_path,
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native

    # Load warm pixels
    poss_warm_pixels = PixelLineCollection()
    poss_warm_pixels.load(dataset.saved_lines(quadrant))
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Figure
    fig = plt.figure(figsize=(10, 9), constrained_layout=False)
    widths = [1, 0.04]
    heights = [1, 1]
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])
    gs.update(wspace=-0.2, hspace=0.02)

    # Zoom regions
    n_row, n_col = image.shape
    col_0 = n_col - int(256 * 3 / 4) + 1
    col_1 = col_0 + int(200 / 3)
    row1_0 = 96
    row1_1 = row1_0 + int(120 / 3)
    row2_0 = n_row - 154
    row2_1 = row2_0 + int(120 / 3)

    # Plot the zooms
    vmin, vmax = 0, 450
    im1 = ax1.imshow(
        X=image[row1_0:row1_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row1_1, row1_0],
    )
    im2 = ax2.imshow(
        X=image[row2_0:row2_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row2_1, row2_0],
    )

    # Warm pixels
    for wp, do_trail in [[poss_warm_pixels, False], [warm_pixels, True]]:
        # Select warm pixels in the regions and above the minimum flux
        rows = wp.locations[:, 0]
        cols = wp.locations[:, 1]
        flux_min = ut.flux_bins[0]
        sel1 = np.where(
            (cols > col_0)
            & (cols < col_1)
            & (rows > row1_0)
            & (rows < row1_1)
            & (wp.fluxes > flux_min)
        )[0]
        sel2 = np.where(
            (cols > col_0)
            & (cols < col_1)
            & (rows > row2_0)
            & (rows < row2_1)
            & (wp.fluxes > flux_min)
        )[0]

        if do_trail:
            c = "#ff1100"
            zorder = 99
        else:
            c = "#ee7711"
            zorder = 88

        # Plot the warm pixels (and trails)
        for ax, sel in [[ax1, sel1], [ax2, sel2]]:
            for i in sel:
                # Outline warm pixel
                c_0 = cols[i] - 0.03
                c_1 = cols[i] + 1.03
                r_0 = rows[i] - 0.03
                r_1 = rows[i] + 1.03
                ax.plot(
                    [c_0, c_1, c_1, c_0, c_0],
                    [r_0, r_0, r_1, r_1, r_0],
                    c=c,
                    lw=0.7,
                    path_effects=[
                        path_effects.Stroke(linewidth=1.4, foreground="k"),
                        path_effects.Normal(),
                    ],
                    zorder=zorder,
                )

                if do_trail:
                    # Outline trail, rounded edges
                    r_0 += 1
                    r_1 += ut.trail_length
                    ax.plot(
                        [c_0, c_0, c_1, c_1],
                        [r_0, r_1, r_1, r_0],
                        c="w",
                        lw=0.7,
                        path_effects=[
                            path_effects.Stroke(linewidth=1.4, foreground="k"),
                            path_effects.Normal(),
                        ],
                    )

    # Axes etc
    cbar = plt.colorbar(im1, cax=cax, extend="max")
    cbar.set_label(r"Flux (e$^-$)")
    ax1.xaxis.set_visible(False)
    ax2.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax2.set_ylabel("Row")
    ax1.set_xlim(col_0, col_1)
    ax2.set_xlim(col_0, col_1)
    ax1.set_ylim(row1_1, row1_0)
    ax2.set_ylim(row2_1, row2_0)
    for ax in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        ax.set_major_locator(mpl.ticker.MultipleLocator(base=10))
        ax.set_minor_locator(mpl.ticker.MultipleLocator(base=5))
    cbar.ax.minorticks_on()
    cbar.ax.yaxis.set_ticks(np.arange(vmin, vmax + 1, 100))
    cbar.ax.yaxis.set_ticks(np.arange(50, vmax + 1, 100), minor=True)
    for ax in [ax1, ax2, cax]:
        set_large_ticks(ax)

    # Save
    save_fig("found_warm_pixels", do_pdf)


def example_single_stack(dataset, do_pdf=False):
    """Example single stack of warm-pixel trails and model fits."""

    # Load
    quadrants = ["A", "B", "C", "D"]
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines(quadrants))
    npzfile = np.load(dataset.saved_stacked_info(quadrants))
    row_bins, flux_bins, date_bins, background_bins = [
        npzfile[var] for var in npzfile.files
    ]
    n_row_bins = len(row_bins) - 1
    n_flux_bins = len(flux_bins) - 1
    n_date_bins = len(date_bins) - 1
    n_background_bins = len(background_bins) - 1

    plt.figure()
    ax = plt.gca()

    # Select the example stack
    bin_index = PixelLineCollection.stacked_bin_index(
        i_row=n_row_bins - 1,
        n_row_bins=n_row_bins,
        i_flux=5,
        n_flux_bins=n_flux_bins,
        i_background=0,
        n_background_bins=n_background_bins,
    )
    line = stacked_lines.lines[bin_index]
    line.remove_symmetry()

    # Don't plot the warm pixel itself
    pixels = np.arange(1, ut.trail_length + 1)
    trail = line.data  # [-ut.trail_length :]
    noise = line.noise  # [-ut.trail_length :]

    # Check for negative values
    where_pos = np.where(trail > 0)[0]
    where_neg = np.where(trail < 0)[0]

    # ========
    # Plot data
    # ========
    c = "k"
    ax.errorbar(
        pixels[where_pos],
        trail[where_pos],
        yerr=noise[where_pos],
        color=c,
        capsize=2,
        alpha=0.7,
        # label=r"$N_{\rm stack} = %d$" % line.n_stacked,
        label="Stacked trail",
    )
    ax.scatter(
        pixels[where_neg],
        abs(trail[where_neg]),
        color=c,
        facecolor="w",
        marker="o",
        alpha=0.7,
        zorder=-1,
    )
    ax.errorbar(
        pixels[where_neg],
        abs(trail[where_neg]),
        yerr=noise[where_neg],
        color=c,
        fmt=",",
        alpha=0.7,
        zorder=-2,
    )

    # ========
    # Fitted exponentials trail
    # ========
    # Fit the total trap density to this single stacked trail
    rho_q, rho_q_std, y_fit = fu.fit_total_trap_density(
        x_all=pixels,
        y_all=trail,
        noise_all=noise,
        n_e_all=line.mean_flux,
        n_bg_all=line.mean_background,
        row_all=line.mean_row,
        date=dataset.date,
    )
    model_pixels = np.linspace(1, ut.trail_length, 20)
    model_trail = fu.trail_model_hst(
        x=model_pixels,
        rho_q=rho_q,
        n_e=line.mean_flux,
        n_bg=line.mean_background,
        row=line.mean_row,
        date=dataset.date,
    )
    ax.plot(
        model_pixels,
        model_trail,
        color=c,
        ls=ls_dash,
        alpha=0.7,
        label=r"$\rho_{\rm q} = %.2f \pm %.2f$" % (rho_q, rho_q_std),
    )
    # Plot each exponential
    if not True:
        for i in range(3):
            # CCD
            beta = 0.478
            w = 84700.0
            # Trap species (one at a time)
            A = 0.17 if i == 0 else 0
            B = 0.45 if i == 1 else 0
            C = 0.38 if i == 2 else 0
            # Trap lifetimes before or after the temperature change
            if line.date < ut.date_T_change:
                tau_a = 0.48
                tau_b = 4.86
                tau_c = 20.6
            else:
                tau_a = 0.74
                tau_b = 7.70
                tau_c = 37.0

            model_i = fu.trail_model(
                x=model_pixels,
                rho_q=rho_q,
                n_e=line.mean_flux,
                n_bg=line.mean_background,
                row=line.mean_row,
                beta=beta,
                w=w,
                A=A,
                B=B,
                C=C,
                tau_a=tau_a,
                tau_b=tau_b,
                tau_c=tau_c,
            )
            ax.plot(model_pixels, model_i, color=c, ls=ls_dot, alpha=0.7)
    #
    # RJM: Jacob's attempt to use arctic instead of exponentials
    #
    #    # ========
    #    # Fitted arctic trail
    #    # ========
    #    density_a, density_b, density_c = fu.fit_trap_densities_arctic(
    #        y_all=[line.data],
    #        noise_all=[line.noise],
    #        n_e_all=[line.mean_flux],
    #        n_bg_all=[line.mean_background],
    #        row_all=[line.mean_row],
    #        date=dataset.date,
    #    )
    #
    #    # Model trail
    #    model_pixels = np.linspace(1, line.mean_row + ut.trail_length, 20)
    #    pre_cti = np.zeros((line.mean_row + ut.trail_length), 1)
    #    pre_cti[line.mean_row, 1] = line.mean_flux
    #    pre_cti += line.mean_background
    #
    #    # trap_densities = np.array([0.17, 0.45, 0.38]) * rho_q
    #    trap_densities = [density_a, density_b, density_c]
    #    if dataset.date < ut.date_T_change:
    #        release_times = np.array([0.48, 4.86, 20.6])
    #    else:
    #        release_times = np.array([0.74, 7.70, 37.0])
    #    roe = cti.ROE(
    #        dwell_times=[1.0],
    #        empty_traps_between_columns=True,
    #        empty_traps_for_first_transfers=False,
    #        force_release_away_from_readout=True,
    #        use_integer_express_matrix=False,
    #    )
    #    ccd = cti.CCD(full_well_depth=84700, well_notch_depth=0.0, well_fill_power=0.478)
    #    traps = [
    #        cti.TrapInstantCapture(
    #            density=trap_densities[i], release_timescale=release_times[i]
    #        )
    #        for i in range(len(trap_densities))
    #    ]
    #
    #    model_trail = cti.add_cti(
    #        image=pre_cti,
    #        parallel_roe=roe,
    #        parallel_ccd=ccd,
    #        parallel_traps=traps,
    #        parallel_express=express,
    #        verbosity=0,
    #    )
    #    ax.plot(
    #        model_pixels,
    #        model_trail,
    #        color=c,
    #        ls=ls_dot,
    #        alpha=0.7,
    #        label=r"$\rho_{\rm q} = %.2f \pm %.2f$" % (rho_q, rho_q_std),
    #    )

    #
    # RJM: temporarily commenting out overlay of results after correction, to avoid having to run the (slow) CTI correction
    #
    #    # ========
    #    # Plot corrected trail
    #    # ========
    #    # Load
    #    stacked_lines_cor = PixelLineCollection()
    #    stacked_lines_cor.load(dataset.saved_stacked_lines(quadrants, use_corrected=True))
    #    line_cor = stacked_lines_cor.lines[bin_index]
    #
    #    # Don't plot the warm pixel itself
    #    trail_cor = line_cor.data[-ut.trail_length :]
    #    noise_cor = line_cor.noise[-ut.trail_length :]
    #
    #    # Check for negative values
    #    where_pos = np.where(trail_cor > 0)[0]
    #    where_neg = np.where(trail_cor < 0)[0]
    #
    #    # Plot
    #    c = "0.4"
    #    ax.errorbar(
    #        pixels[where_pos],
    #        trail_cor[where_pos],
    #        yerr=noise_cor[where_pos],
    #        color=c,
    #        capsize=2,
    #        alpha=0.7,
    #        label="After correction",
    #    )
    #    ax.scatter(
    #        pixels[where_neg],
    #        abs(trail_cor[where_neg]),
    #        color=c,
    #        facecolor="w",
    #        marker="o",
    #        alpha=0.7,
    #        zorder=-1,
    #    )
    #    ax.errorbar(
    #        pixels[where_neg],
    #        abs(trail_cor[where_neg]),
    #        yerr=noise_cor[where_neg],
    #        color=c,
    #        fmt=",",
    #        alpha=0.7,
    #        zorder=-2,
    #    )

    # ========
    # Axes etc
    # ========
    ax.set_xlim(0.5, ut.trail_length + 0.5)
    ax.set_xticks(np.arange(1, ut.trail_length + 0.1, 1))
    ax.set_xlabel(r"Row relative to warm pixel")
    ax.set_ylabel("Charge above background (e$^-$)")
    ax.set_yscale("log")
    ax.legend()

    nice_plot()

    # Save
    save_fig("example_single_stack", do_pdf)


def example_stacked_trails(dataset, do_pdf=False):
    """Example stacked trails in bins."""

    # Plot
    fu.plot_stacked_trails(dataset, quadrants=["A", "B", "C", "D"], save_path="None")

    # Save
    save_fig("example_stacked_trails", do_pdf)


def example_stacked_trails(dataset, do_pdf=False):
    """Example stacked trails in bins, using arCTIc."""

    # Plot
    fu.plot_stacked_trails(dataset, quadrants=["A", "B", "C", "D"], save_path="None")

    # Save
    save_fig("example_stacked_arctic_trails", do_pdf)


def density_evol(do_pdf=False):
    """Evolution of the total trap density."""

    list_name = "sample"
    quadrant_sets = [["A", "B", "C", "D"]]

    # Plot
    fu.plot_trap_density_evol(
        list_name, quadrant_sets, do_sunspots=True, use_corrected=True, do_pdf=do_pdf
    )


# ========
# Main
# ========
if __name__ == "__main__":
    # Parse arguments
    parser = prep_parser()
    args = parser.parse_args()

    # Example dataset and image
    dataset = Dataset("07_2020")  # 2020/07/31, day 6727, 8 images
    image_name = "jdrwc3fcq_raw"
    quadrant = "D"
    n_iterations = 1
    use_corrected = False

    # Where to find the example data (dataset.path is defined in hst_utilities.py)
    image_path = dataset.path + image_name + ".fits"
    cor_path = dataset.path + image_name + "_cor_iter%d.fits" % n_iterations

    # Run functions
    # if run("example_image_zooms"):
    #    example_image_zooms( image_path, cor_path, quadrant, args.pdf, use_corrected )
    # if run("example_image_corrected"):
    #    example_image_corrected(image_path, cor_path, quadrant, n_iterations, args.pdf )
    # if run("found_warm_pixels"):
    #    found_warm_pixels( image_path, quadrant, args.pdf )
    if run("example_single_stack"):
        example_single_stack(dataset, args.pdf)
    if run("example_stacked_trails"):
        example_stacked_trails(dataset, args.pdf)
    if run("example_stacked_trails"):
        example_stacked_trails(dataset, args.pdf)
    # if run("density_evol"):
    #    density_evol(args.pdf)
    # print("done6")
    # if run("test_autofit_model"):
    #    test_autofit_model(args.pdf)
