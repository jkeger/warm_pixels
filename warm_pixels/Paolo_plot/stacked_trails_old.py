import logging
#from pathlib import Path
#import warm_pixels as wp
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import autofit as af
from warm_pixels import hst_utilities as ut#, PixelLine
from warm_pixels import misc
#from warm_pixels.hst_functions.fit import fit_dataset_total_trap_density
#from warm_pixels.hst_functions.trail_model import trail_model_hst
from warm_pixels.hst_functions.trail_model import trail_model # Use the version with k constant
#from warm_pixels.fit.model import TrailModel
#from warm_pixels.fit.analysis import Analysis
from warm_pixels.model.group import QuadrantGroup
from autoarray.fit.fit_dataset import SimpleFit

logger = logging.getLogger(
    __name__
)


def plot_stacked_trails(group: QuadrantGroup, use_corrected=False, save_path=None):
    class Analysis(af.Analysis):
        def __init__(self, x, y, noise, n_e, n_bg, row):
            self.x = x
            self.y = y
            self.noise = noise
            self.n_e = n_e
            self.n_bg = n_bg
            self.row = row
    
        def visualize(self, paths, instance, during_analysis):
            #plt.plot(self.x, self.y)
            #plt.plot(self.x, instance(
            #    x=self.x,
            #    n_e=self.n_e,
            #    n_bg=self.n_bg,
            #    row=self.row,
            #)
            #)
            print('Visualising')
    
        def log_likelihood_function(self, instance):
            modelled_trail = instance(
                x=self.x,
                n_e=self.n_e,
                n_bg=self.n_bg,
                row=self.row,
            )
            fit = SimpleFit(
                data=self.y,
                model_data=modelled_trail,
                noise_map=self.noise,
            )
            return fit.log_likelihood
        
    class TrailModel:
        def __init__(
                self,
                rho_q,
                beta,
                w,
                a,
                b,
                c,
                tau_a,
                tau_b,
                tau_c,
        ):
            self.rho_q = rho_q
            self.beta = beta
            self.w = w
            self.a = a
            self.b = b
            self.c = c
            self.tau_a = tau_a
            self.tau_b = tau_b
            self.tau_c = tau_c
    
        def __call__(self, x, n_e, n_bg, row):
            return trail_model(
                x=x,
                rho_q=self.rho_q,
                n_e=n_e,
                n_bg=n_bg,
                row=row,
                beta=self.beta,
                w=self.w,
                A=self.a,
                B=self.b,
                C=self.c,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                tau_c=self.tau_c,
            )
    
    stacked_lines = group.stacked_lines()
    
    #date = stacked_lines.date How can I get the date value from stacked_lines? 
    
    # Define constants and free variables
    # CCD
    beta = 0.478
    w = 84700.0
    # Trap species
    a = 0.17
    b = 0.45
    c = 0.38
    
    # Trap lifetimes before or after the temperature change
# =============================================================================
#     if date < ut.date_T_change:
#         tau_a = 0.48
#         tau_b = 4.86
#         tau_c = 20.6
#     else:
#         tau_a = 0.74
#         tau_b = 7.70
#         tau_c = 37.0
#     
# =============================================================================
    tau_a = 0.74
    tau_b = 7.70
    tau_c = 37.0
    
    # CCD
    rho_q = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=10.0,
    )
    beta = af.GaussianPrior(
        mean=0.478,
        sigma=0.1,
    )
    # w = af.GaussianPrior(
    #     mean=84700.0,
    #     sigma=20000,
    # )
    
    # Trap species
    a = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=1.0,
    )
    b = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=1.0,
    )
    c = 1 - (a + b)
    
    tau_a = af.GaussianPrior(
          mean=tau_a,
          sigma=0.2,
      )
    tau_b = af.GaussianPrior(
          mean=tau_b,
          sigma=2.0,
      )
    tau_c = af.GaussianPrior(
          mean=tau_c,
          sigma=10.0,
      )
    
    model = af.Model(
        TrailModel,
        rho_q=rho_q,
        beta=beta,
        w=w,
        a=a,
        b=b,
        c=c,
        tau_a=tau_a,
        tau_b=tau_b,
        tau_c=tau_c,
    )
    
    model.add_assertion(c > 0.0)
    
    # Extract row bins
    n_row_bins = stacked_lines.n_row_bins
    n_flux_bins = stacked_lines.n_flux_bins
    n_background_bins = stacked_lines.n_background_bins

    # Plot the stacked trails
    plt.figure(figsize=(25, 12))
    gs = GridSpec(n_row_bins, n_flux_bins)
    axes = [
        [plt.subplot(gs[i_row, i_flux]) for i_flux in range(n_flux_bins)]
        for i_row in range(n_row_bins)
    ]
    gs.update(wspace=0, hspace=0)

    # Don't plot the warm pixel itself
    pixels = np.arange(1, ut.trail_length + 1)
    sel_non_zero = np.where(stacked_lines.data[:, -ut.trail_length:] != 0)
    # Set y limits
    if use_corrected:
        # For symlog scale
        # Assume ymin < 0
        y_min = 0.1  # 4 * np.amin(stacked_lines.data[:, -ut.trail_length :][sel_non_zero])
        y_max = 4 * np.amax(stacked_lines.data[:, -ut.trail_length:][sel_non_zero])
        log10_y_min = np.ceil(np.log10(abs(y_min)))
        log10_y_max = np.floor(np.log10(y_max))
        y_min = min(y_min, -10 ** (log10_y_min + 0.6))
        y_max = max(y_max, 10 ** (log10_y_max + 0.6))
        y_ticks = np.append(
            -10 ** np.arange(log10_y_min, -0.1, -1),
            10 ** np.arange(0, log10_y_max + 0.1, 1),
        )
    else:
        # For log scale
        y_min = np.partition(
            abs(np.ravel(stacked_lines.data[:, -ut.trail_length:][sel_non_zero])), 2
        )[1]
        y_min = 0.1
        y_max = 4 * np.amax(stacked_lines.data[:, -ut.trail_length:][sel_non_zero])
        log10_y_min = np.ceil(np.log10(y_min))
        log10_y_max = np.floor(np.log10(y_max))
        y_min = min(y_min, 10 ** (log10_y_min - 0.4))
        y_max = max(y_max, 10 ** (log10_y_max + 0.4))
        y_ticks = 10 ** np.arange(log10_y_min, log10_y_max + 0.1, 1)
    if n_background_bins == 1:
        colours = ["k"]
    else:
        colours = plt.cm.jet(np.linspace(0.05, 0.95, n_background_bins))

    # Label size
    fontsize = 14

    # Fit the total trap density to the full dataset
# =============================================================================
#     print("Performing global LMFIT")
#     rho_q_set, rho_q_std_set, y_fit = fit_dataset_total_trap_density(
#         group, use_arctic=False
#     )
# =============================================================================
    #print(rho_q_set, rho_q_std_set, "exponentials")

    # Fit the total trap density to the full dataset using arCTIc
    # print("Performing global fit")
    # rho_q_set, rho_q_std_set, y_fit = fit_dataset_total_trap_density(
    #     dataset, quadrants, use_corrected=use_corrected, use_arctic=True
    # )
    # print(rho_q_set, rho_q_std_set, "ArCTIc")

    # Fit the total trap density to the full dataset using arCTIc and MCMC
    # print("Performing global fit using arCTIc")
    # result = fit_warm_pixels_with_arctic(
    #    dataset, quadrants, use_corrected=use_corrected
    # )

    
    # Compile the data from all stacked lines for the global AUTOFIT
    print('Extracting data for global autofit')
    n_lines_used = 0
    y_all = np.array([])
    #x_all = np.array([])
    noise_all = np.array([])
    n_e_each = np.array([])
    n_bg_each = np.array([])
    row_each = np.array([])
   #x_one=np.array(np.arange(ut.trail_length)+1)

    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            ax = axes[n_row_bins - 1 - i_row][i_flux]
            for i_background, c in enumerate(colours):
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0)
                
                if line.n_stacked >= 3:
                    
                    # Compile data into easy form to fit
                    y_most_neg_each=min(line.model_trail)
                    y_all = np.append(y_all, np.array(line.model_trail)+abs(y_most_neg_each)) # add the most negative value
                    noise_all = np.append(noise_all, np.array(line.model_trail_noise))
                    #n_e_all=np.append(n_e_each,np.array(np.repeat(line.model_flux,ut.trail_length)))
                    #n_bg_all=np.append(n_bg_each,np.array(np.repeat(line.model_background,ut.trail_length)))
                    #row_all=np.append(n_bg_each,np.array(np.repeat(line.mean_row,ut.trail_length)))
                    #x_all=np.append(x_all,x_one)
                    #n_e_each = np.append(n_e_each, line.model_flux)
                    #n_bg_each = np.append(n_bg_each, line.model_background)
                    n_e_each = np.append(n_e_each, line.mean_flux)
                    n_bg_each = np.append(n_bg_each, line.mean_background)
                    row_each = np.append(row_each, line.mean_row)
                    n_lines_used += 1
    if n_lines_used == 0:
        return None, None, np.zeros(ut.trail_length)
    
    # Find the most negative trail value from the combined 50 plots and add it to all trail values
# =============================================================================
#     print('Checking trails for negative values')
#     y_most_neg=min(y_all)
#     if y_most_neg<0:
#         print('Smallest value found: ', y_most_neg)
#         y_all=y_all+abs(y_most_neg)
#     else: print('No negative values found')
# =============================================================================

    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(ut.trail_length) + 1, n_lines_used)
    #x_one=np.array(np.arange(ut.trail_length)+1)

    # Duplicate the single parameters of each trail for all pixels
    n_e_all = np.repeat(n_e_each, ut.trail_length)
    n_bg_all = np.repeat(n_bg_each, ut.trail_length)
    row_all = np.repeat(row_each, ut.trail_length)
    
    # Make instance of analysis, passing it the data.  
    analysis = Analysis(
       x=x_all,
       y=y_all,
       noise=noise_all,
       n_e=n_e_all,
       n_bg=n_bg_all,
       row=row_all,
    )
    
    #plt.plot(analysis.x, analysis.y, label='Analysis x and y')
    
    # Load our optimiser
    dynesty = af.DynestyStatic()
    
    # Do the fitting
    print('Perfoming global AUTOFIT: ')
    result = dynesty.fit(
    model=model,
    analysis=analysis,
    )
    
    print(f"log likelihood = {result.log_likelihood}")
    
    best_trail_model = result.instance

    print(f"rho_q = {best_trail_model.rho_q}")
    print(f"a = {best_trail_model.a}")
    print(f"b = {best_trail_model.b}")
    print(f"c = {best_trail_model.c}")
    print(f"tau_a = {best_trail_model.tau_a}")
    print(f"tau_b = {best_trail_model.tau_b}")
    print(f"tau_c = {best_trail_model.tau_c}")
    
    print('Generating plots')
    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            # Furthest row bin at the top
            ax = axes[n_row_bins - 1 - i_row][i_flux]

            # Plot each background bin's stack
            for i_background, c in enumerate(colours):
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0
                )
                # Skip empty and single-entry bins
                if line.n_stacked <= 1:
                    continue
                
                # Don't plot the warm pixel itself
                y_most_neg_local = min(line.model_trail)
                trail = np.array(line.model_trail) + abs(y_most_neg_local)   # + line.model_background
                noise = line.model_trail_noise  # + line.model_background

                # Check for negative values
                where_pos = np.where(trail > 0)[0]
                where_neg = np.where(trail < 0)[0]

                # ========
                # Plot data
                # ========
                if use_corrected:
                    # Plot positives and negatives together for symlog scale
                    ax.errorbar(
                        pixels, trail, yerr=noise, color=c, capsize=2, alpha=0.7
                    )
                else:
                    # Plot positives and negatives separately for log scale
                    ax.errorbar(
                        pixels[where_pos],
                        trail[where_pos],
                        yerr=noise[where_pos],
                        color=c,
                        capsize=2,
                        alpha=0.7,
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
                # Plot fitted trail
                # ========
                # Fit the total trap density to this single stacked trail (dotted line, which has swapped since Jacob's version)
                
                #print('Performing individual LMFITS')
# =============================================================================
#                 rho_q_indiv, rho_q_std_indiv, y_fit_indiv = fit_dataset_total_trap_density(
#                     group, use_arctic=True,
#                     row_bins=[i_row], flux_bins=[i_flux], background_bins=[i_background]
#                 )
#                 ax.plot(pixels, y_fit_indiv, color=c, ls=misc.ls_dot, alpha=0.7)
# =============================================================================
                
                # Also reconstruct then plot the simultaneous fit to all trails (dashed line)
# =============================================================================
#                 model_trail = trail_model_hst(
#                     x=pixels,
#                     rho_q=rho_q_set,
#                     n_e=line.mean_flux,
#                     n_bg=line.mean_background,
#                     row=line.mean_row,
#                     date=group.dataset.date,
#                 )
#                 ax.plot(pixels, model_trail, color='red', ls=misc.ls_dash, alpha=0.7)
# =============================================================================
                
                # Plot the global autofit model 
                global_autofit=trail_model(x=pixels, 
                                           rho_q=float(best_trail_model.rho_q), 
                                           n_e=float(line.mean_flux), 
                                           n_bg=float(line.mean_background), 
                                          # n_e=line.model_flux, 
                                           #n_bg=line.model_background, 
                                           row=float(line.mean_row), 
                                           beta=float(beta), 
                                           w=float(w), 
                                           A=float(best_trail_model.a), 
                                           B=float(best_trail_model.b), 
                                           C=float(best_trail_model.c), 
                                           tau_a=float(best_trail_model.tau_a), 
                                           tau_b=float(best_trail_model.tau_b), 
                                           tau_c=float(best_trail_model.tau_c)
                                          )

                ax.plot(pixels, global_autofit, color='green', ls='-.', alpha=0.7)
                # Annotate
                if i_background == 0:
                    text = "$%d$" % line.n_stacked
                else:
                    text = "\n" * i_background + "$%d$" % line.n_stacked
                ax.text(
                    0.97,
                    0.96,
                    text,
                    transform=ax.transAxes,
                    size=fontsize,
                    ha="right",
                    va="top",
                )

            ax.set_xlim(0.5, ut.trail_length + 0.5)
            ax.set_xticks(np.arange(2, ut.trail_length + 0.1, 2))
            ax.set_xticks(np.arange(1, ut.trail_length + 0.1, 2), minor=True)
            if use_corrected:
                ax.set_yscale("symlog", linthreshy=1, linscaley=0.5)
                ax.axhline(0, lw=0.5, c="0.7", zorder=-99)
            else:
                ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)

            # Axis labels
            if i_row != 0:
                ax.set_xticklabels([])
            elif i_flux in [2, n_flux_bins - 3]:
                ax.set_xlabel("Pixel")
            if i_flux != 0:
                ax.set_yticklabels([])
            elif i_row in [1, n_row_bins - 2]:
                ax.set_ylabel("Number of electrons (e$^-$)")

            # Bin edge labels
            if i_flux == n_flux_bins - 1:
                if i_row == 0:
                    ax.text(
                        1.02,
                        0.5,
                        "Row:",
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
                if i_row < n_row_bins - 1:
                    ax.text(
                        1.02,
                        1.0,
                        "%d" % stacked_lines.row_bins[i_row + 1],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
            if i_row == n_row_bins - 1:
                if i_flux == 0:
                    ax.text(
                        0.5,
                        1.01,
                        r"e$^-$ flux:",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )
                flux_max = stacked_lines.flux_bins[i_flux + 1]
                pow10 = np.floor(np.log10(flux_max))
                text = r"$%.1f \!\times\! 10^{%d}$" % (flux_max / 10 ** pow10, pow10)
                ax.text(
                    1.0, 1.01, text, transform=ax.transAxes, ha="center", va="bottom"
                )
            if i_row == int(n_row_bins / 2) and i_flux == n_flux_bins - 1:
                text = "Background:  "
                for i_background in range(n_background_bins):
                    text += "%.0f$-$%.0f" % (
                        stacked_lines.background_bins[i_background],
                        stacked_lines.background_bins[i_background + 1],
                    )
                    if i_background < n_background_bins - 1:
                        text += ",  "
                ax.text(
                    1.25,
                    0.5,
                    text,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="left",
                    va="center",
                )

            # Total trap density
# =============================================================================
#             if i_row == n_row_bins - 1 and i_flux == n_flux_bins - 1:
#                 if rho_q_set is None or rho_q_std_set is None:
#                     text = "fit error"
#                 else:
#                     text = r"$\rho_{\rm q} = %.3f \pm %.3f$" % (
#                         rho_q_set,
#                         rho_q_std_set,
#                     )
# =============================================================================
# =============================================================================
#                 ax.text(
#                     0.03,
#                     0.03,
#                     text,
#                     transform=ax.transAxes,
#                     size=fontsize,
#                     ha="left",
#                     va="bottom",
#                 )
# =============================================================================

            # Tidy
            if i_row == 0 and i_flux == 0:
                misc.set_large_ticks(ax)
            elif i_row == 0:
                misc.set_large_ticks(ax, do_y=False)
            elif i_flux == 0:
                misc.set_large_ticks(ax, do_x=False)
            misc.set_font_size(ax)

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path.name)

