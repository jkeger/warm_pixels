import logging
from pathlib import Path
import warm_pixels as wp
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import autofit as af
from warm_pixels import hst_utilities as ut#, PixelLine
from warm_pixels import misc
#from warm_pixels.hst_functions.fit import fit_dataset_total_trap_density
#from warm_pixels.hst_functions.trail_model import trail_model_hst
#from warm_pixels.hst_functions.trail_model_k_fastest import trail_model_arctic_continuum_notch 
#from warm_pixels.fit.model import TrailModel
#from warm_pixels.fit.analysis import Analysis
from warm_pixels.model.group import QuadrantGroup
from autoarray.fit.fit_dataset import SimpleFit
import time
import csv
from os import path
import sys


cosma_id = int(sys.argv[1])

logger = logging.getLogger(
    __name__
)

start_time = time.time()
def trail_model_exp(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, notch):
    
    """Calculate the model shape of a CTI trail.

    Parameters
    ----------
    x : [float]
        The pixel positions away from the trailed pixel.

    rho_q : float
        The total trap number density per pixel.

    n_e : float
        The number of electrons in the trailed pixel's charge cloud (e-).

    n_bg : float
        The background number of electrons (e-).

    row : float
        The distance in pixels of the trailed pixel from the readout register.

    beta : float
        The CCD well fill power.

    w : float
        The CCD full well depth (e-).

    A, B, C : float
        The relative density of each trap species.

    tau_a, tau_b, tau_c : float
        The release timescale of each trap species (s).

    Returns
    -------
    trail : [float]
        The model charge values at each pixel in the trail (e-).
    """
    # print(n_bg,n_e)
    
    
    #print('first term denominator =', (w - notch))
    local_counter=0
    local_array=[]
    
    #print('len n_bg is', len(n_bg))
    while local_counter<len(n_bg):
# =============================================================================
#         term1=np.abs(n_e[local_counter]) - notch
#         #print('term1 is', term1)
#         term2=np.abs(n_bg[local_counter]) - notch
#         #print('term2 is', term2)
# =============================================================================
        volume1 = np.sign(n_e[local_counter]) * np.clip((abs(n_e[local_counter]) - notch) / (w - notch), 0, 1) ** beta
        volume2 = np.sign(n_bg[local_counter]) * np.clip((abs(n_bg[local_counter]) - notch) / (w - notch), 0, 1) ** beta
        local_array.append(
                rho_q
        * (volume1 - volume2)
        * row[local_counter]
        * (
                A * np.exp((1 - x[local_counter]) / tau_a) * (1 - np.exp(-1 / tau_a))
                + B * np.exp((1 - x[local_counter]) / tau_b) * (1 - np.exp(-1 / tau_b))
                + C * np.exp((1 - x[local_counter]) / tau_c) * (1 - np.exp(-1 / tau_c))
        )  
        )
        local_counter=local_counter+1
    return (local_array)
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
            print('log likelihood = ', fit.log_likelihood)
            return fit.log_likelihood
        
class TrailModelPrint:
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
                notch
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
            self.notch=notch
    
        def __call__(self, x, n_e, n_bg, row):
            return trail_model_exp(
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
                notch=self.notch,
            )
        
# Now do the 50 plot diagram for the corrected dataset and it's best fit model.
def Paolo_autofit_global_50_after(group1: QuadrantGroup, group2: QuadrantGroup, group3: QuadrantGroup,
                                  use_corrected=False, save_path=None): 
    
    stacked_lines1 = group1.stacked_lines()
    stacked_lines2 = group2.stacked_lines()
    #stacked_lines3 = group3.stacked_lines()
    
    #date = stacked_lines.date How can I get the date value from stacked_lines? 
    
    # Define constants and free variables
    # CCD
    beta = 0.478
    w = 84700.0
    # Trap species
    a = 0.17
    b = 0.45
    c = 0.38
   
    
    # Extract row bins
    n_row_bins1 = stacked_lines1.n_row_bins
    n_flux_bins1 = stacked_lines1.n_flux_bins
    n_background_bins1 = stacked_lines1.n_background_bins
    
    n_row_bins2 = stacked_lines2.n_row_bins
    n_flux_bins2 = stacked_lines2.n_flux_bins
    n_background_bins2 = stacked_lines2.n_background_bins

    # Plot the stacked trails
    plt.figure(figsize=(25, 12))
    gs = GridSpec(n_row_bins1, n_flux_bins1)
    axes = [
        [plt.subplot(gs[i_row, i_flux]) for i_flux in range(n_flux_bins1)]
        for i_row in range(n_row_bins1)
    ]
    gs.update(wspace=0, hspace=0)

    # Don't plot the warm pixel itself
    pixels = np.arange(1, ut.trail_length + 1)
    sel_non_zero = np.where(stacked_lines1.data[:, -ut.trail_length:] != 0)
    # Set y limits
    if use_corrected:
        # For symlog scale
        # Assume ymin < 0
        y_min = 0.1  # 4 * np.amin(stacked_lines.data[:, -ut.trail_length :][sel_non_zero])
        y_max = 4 * np.amax(stacked_lines1.data[:, -ut.trail_length:][sel_non_zero])
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
            abs(np.ravel(stacked_lines1.data[:, -ut.trail_length:][sel_non_zero])), 2
        )[1]
        y_min = 0.1
        y_max = 4 * np.amax(stacked_lines1.data[:, -ut.trail_length:][sel_non_zero])
        log10_y_min = np.ceil(np.log10(y_min))
        log10_y_max = np.floor(np.log10(y_max))
        y_min = min(y_min, 10 ** (log10_y_min - 0.4))
        y_max = max(y_max, 10 ** (log10_y_max + 0.4))
        y_ticks = 10 ** np.arange(log10_y_min, log10_y_max + 0.1, 1)
    if n_background_bins1 == 1:
        colours = ["k"]
    else:
        colours = plt.cm.jet(np.linspace(0.05, 0.95, n_background_bins1))

    # Label size
    fontsize = 20

    
# =============================================================================
#     # Compile the data from all stacked lines for the global AUTOFIT
#     n_lines_used = 0
#     y_all = np.array([])
#     #x_all = np.array([])
#     N_each = np.array([])
#     noise_all = np.array([])
#     n_e_each = np.array([])
#     n_bg_each = np.array([])
#     row_each = np.array([])
#    #x_one=np.array(np.arange(ut.trail_length)+1)
#    
#     
#     for i_row in range(n_row_bins):
#         for i_flux in range(n_flux_bins):
#             ax = axes[n_row_bins - 1 - i_row][i_flux]
#             for i_background, c in enumerate(colours):
#                 line = stacked_lines.stacked_line_for_indices(
#                     row_index=i_row,
#                     flux_index=i_flux,
#                     background_index=i_background,
#                     date_index=0)
#                 
#                 if line.n_stacked >= 3:
# 
#                     #
#                     # Compile data into easy form to fit
#                     #
#                     
#                     N_local=min(line.model_trail) # identify least value
#                     if N_local<0: # add only if negative
#                         y_all = np.append(y_all, np.array(line.model_trail)+abs(N_local))
#                         N_each = np.append(N_each, abs(N_local))
#                     else:
#                         y_all = np.append(y_all, np.array(line.model_trail))
#                         N_each = np.append(N_each, 0)
#                     noise_all = np.append(noise_all, np.array(line.model_trail_noise))
#                     n_e_each = np.append(n_e_each, line.mean_flux)
#                     n_bg_each = np.append(n_bg_each, line.mean_background)
#                     row_each = np.append(row_each, line.mean_row)
#                     n_lines_used += 1
#     if n_lines_used == 0:
#         return None, None, np.zeros(ut.trail_length)
# 
#     # Duplicate the x arrays for all trails
#     x_all = np.tile(np.arange(ut.trail_length) + 1, n_lines_used)
#     #x_one=np.array(np.arange(ut.trail_length)+1)
# 
#     # Duplicate the single parameters of each trail for all pixels
#     n_e_all = np.repeat(n_e_each, ut.trail_length)
#     n_bg_all = np.repeat(n_bg_each, ut.trail_length)
#     row_all = np.repeat(row_each, ut.trail_length)
#     #N_all = np.repeat(N_each, ut.trail_length)
# =============================================================================
    
    # Make instance of analysis, passing it the data.  
# =============================================================================
#     analysis = Analysis(
#        x=x_all,
#        y=y_all,
#        noise=noise_all,
#        n_e=n_e_all,
#        n_bg=n_bg_all,
#        row=row_all
#     )
# =============================================================================
    
    #plt.plot(analysis.x, analysis.y, label='Analysis x and y')
    
    # Load our optimiser
    #dynesty = af.DynestyStatic(number_of_cores=16, sample="rwalk", walks=5, iterations_per_update=10000000)#, #force_x1_cpu=True)
    
    #print(dynesty.config_dict_run)
    #exit(dynesty.config_dict_search)
    
    # Do the fitting
# =============================================================================
#     print('Perfoming global AUTOFIT: ')
#     result = dynesty.fit(
#     model=model,
#     analysis=analysis,
#     )
# =============================================================================
    
# =============================================================================
#     print(f"log likelihood = {result.log_likelihood}")
#     
#     best_trail_model = result.instance
#     print(result.info)
# 
#     print(f"beta = {best_trail_model.beta}")
#     print(f"rho_q = {best_trail_model.rho_q}")
#     print(f"a = {best_trail_model.a}")
#     print(f"b = {best_trail_model.b}")
#     print(f"c = {best_trail_model.c}")
#     print(f"tau_a = {best_trail_model.tau_a}")
#     print(f"tau_b = {best_trail_model.tau_b}")
#     print(f"tau_c = {best_trail_model.tau_c}")
#     print(f"sigma_a = {best_trail_model.sigma_a}")
#     print(f"sigma_b = {best_trail_model.sigma_b}")
#     print(f"sigma_c = {best_trail_model.sigma_c}")
#     print(f"notch = {best_trail_model.notch}")
# =============================================================================
    
    line_means=[]
    mean_mean=[]
    for i_row in range(n_row_bins1):
        for i_flux in range(n_flux_bins1):
            # Furthest row bin at the top
            ax = axes[n_row_bins1 - 1 - i_row][i_flux]

            # Plot each background bin's stack
            for i_background, c in enumerate(colours):
                line1 = stacked_lines1.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0
                )
                line2 = stacked_lines2.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0
                )
# =============================================================================
#                 line3 = stacked_lines3.stacked_line_for_indices(
#                     row_index=i_row,
#                     flux_index=i_flux,
#                     background_index=i_background,
#                     date_index=0
#                 )
# =============================================================================
                # Skip empty and single-entry bins
                if line1.n_stacked <= 1:
                    continue
                
                # Don't plot the warm pixel itself
                
# =============================================================================
#                 N_local=min(line1.model_trail)
#                 #N_plotted=0
#                 if N_local<0:
#                     trail1 = np.array(line1.model_trail)+abs(N_local)
#                     
#                     #N_plotted = abs(N_local)
#                 else:
# =============================================================================
                trail1 = line1.model_trail  # + line.model_background
                    
                noise1 = line1.model_trail_noise  # + line.model_background
                
                

                # Check for negative values
                where_pos1 = np.where(trail1 > 0)[0]
                where_neg1 = np.where(trail1 < 0)[0]
                
                
                trail2 = line2.model_trail  # + line.model_background
                    
                noise2 = line2.model_trail_noise  # + line.model_background
                
                

                # Check for negative values
                where_pos2 = np.where(trail2 > 0)[0]
                where_neg2 = np.where(trail2 < 0)[0]

                # Don't plot the warm pixel itself
                

                

                
# =============================================================================
#                 # ========
#                 # Plot data
#                 # ========
#                 if use_corrected:
#                     # Plot positives and negatives together for symlog scale
# # =============================================================================
# #                     ax.errorbar(
# #                         pixels, trail1, yerr=noise1, color='red', capsize=2, alpha=0.7
# #                     )
# # =============================================================================
# # =============================================================================
# #                     ax.errorbar(
# #                         pixels, trail2, yerr=noise2, color='green', capsize=2, alpha=0.7
# #                     )
# # =============================================================================
#                     
#                 else:
# =============================================================================
                # Plot positives and negatives separately for log scale
                ax.errorbar(
                    pixels[where_pos1],
                    trail1[where_pos1],
                    yerr=noise1[where_pos1],
                    color='red',
                    capsize=2,
                    alpha=0.7,
                )
                ax.errorbar(
                    pixels[where_pos2],
                    trail2[where_pos2],
                    yerr=noise2[where_pos2],
                    color='green',
                    capsize=2,
                    alpha=0.7,
                )
# =============================================================================
#                 ax.errorbar(
#                     pixels[where_pos3],
#                     trail3[where_pos3],
#                     yerr=noise3[where_pos3],
#                     color='black',
#                     capsize=2,
#                     alpha=0.7,
#                 )
# =============================================================================
                ax.scatter(
                    pixels[where_neg1],
                    abs(trail1[where_neg1]),
                    color=c,
                    facecolor="w",
                    marker="o",
                    alpha=0.7,
                    zorder=-1,
                )
                ax.errorbar(
                    pixels[where_neg1],
                    abs(trail1[where_neg1]),
                    yerr=noise1[where_neg1],
                    color='red',
                    fmt=",",
                    alpha=0.7,
                    zorder=-2,
                )
                ax.scatter(
                    pixels[where_neg2],
                    abs(trail2[where_neg2]),
                    color=c,
                    facecolor="w",
                    marker="o",
                    alpha=0.7,
                    zorder=-1,
                )
                ax.errorbar(
                    pixels[where_neg2],
                    abs(trail2[where_neg2]),
                    yerr=noise2[where_neg2],
                    color='green',
                    fmt=",",
                    alpha=0.7,
                    zorder=-2,
                )
                
                #ax.plot(pixels, global_autofit, color='green', ls='-.', alpha=0.7)
                # Annotate
# =============================================================================
#                 if i_background == 0:
#                     text = "$%d$" % line1.n_stacked
#                 else:
#                     text = "\n" * i_background + "$%d$" % line.n_stacked
# =============================================================================
# =============================================================================
#                 ax.text(
#                     0.97,
#                     0.96,
#                     text,
#                     transform=ax.transAxes,
#                     size=fontsize,
#                     ha="right",
#                     va="top",
#                 )
# =============================================================================

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
            elif i_flux in [2, n_flux_bins1 - 3]:
                ax.set_xlabel("Pixel")
            if i_flux != 0:
                ax.set_yticklabels([])
            elif i_row in [1, n_row_bins1 - 2]:
                ax.set_ylabel("Number of electrons (e$^-$)")

            # Bin edge labels
            if i_flux == n_flux_bins1 - 1:
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
                if i_row < n_row_bins1 - 1:
                    ax.text(
                        1.02,
                        1.0,
                        "%d" % stacked_lines1.row_bins[i_row + 1],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
            if i_row == n_row_bins1 - 1:
                if i_flux == 0:
                    ax.text(
                        0.3,
                        1.01,
                        r"e$^-$ Flux:",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )
                flux_max = stacked_lines1.flux_bins[i_flux + 1]
                pow10 = np.floor(np.log10(flux_max))
                text = r"$%.1f \!\times\! 10^{%d}$" % (flux_max / 10 ** pow10, pow10)
                ax.text(
                    1.0, 1.01, text, transform=ax.transAxes, ha="center", va="bottom"
                )
            if i_row == int(n_row_bins1 / 2) and i_flux == n_flux_bins1 - 1:
                text = "Background (e$^-$):  "
                for i_background in range(n_background_bins1):
                    text += "%.0f$-$%.0f" % (
                        stacked_lines1.background_bins[i_background],
                        stacked_lines1.background_bins[i_background + 1],
                    )
                    if i_background < n_background_bins1 - 1:
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

            # Tidy
            if i_row == 0 and i_flux == 0:
                misc.set_large_ticks(ax)
            elif i_row == 0:
                misc.set_large_ticks(ax, do_y=False)
            elif i_flux == 0:
                misc.set_large_ticks(ax, do_x=False)
            misc.set_font_size(ax)

    plt.tight_layout()
    mean_mean.append(np.mean(line_means))
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path.name)
        
    print("Total post correction fit processing time: ", time.time() - start_time, "seconds")
    

# Import data to be fitted
cosma_dataset_path = path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", "stock_exp_free_c_1.0", "07_2011_stock_exp_free_c_1.0")
cosma_path = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6")
cosma_output_path = path.join(cosma_path, "output")
workspace_path = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"
#config_path = path.join(workspace_path, "cosma", "config")

dataset_directory=Path(cosma_dataset_path)


dataset = wp.Dataset(dataset_directory)

group1 = dataset.group("ABCD")

cosma_dataset_path = path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", "stock_exp_free_v2_1.0", "07_2011_stock_exp_free_v2_1.0")
cosma_path = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6")
cosma_output_path = path.join(cosma_path, "output")
workspace_path = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"
#config_path = path.join(workspace_path, "cosma", "config")

dataset_directory=Path(cosma_dataset_path)


dataset = wp.Dataset(dataset_directory)

group2 = dataset.group("ABCD")

cosma_path = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6")
#dataset_folder="Paolo's_03_2020"
#dataset_name="03_2020"

cosma_dataset_path = path.join(cosma_path, "dataset", "03_2020")
cosma_output_path = path.join(cosma_path, "output")
workspace_path = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"
#config_path = path.join(workspace_path, "cosma", "config")

dataset_directory=Path(cosma_dataset_path)


dataset = wp.Dataset(dataset_directory)

group3 = dataset.group("ABCD")


# Call the 50 plot function we just defined    
Paolo_autofit_global_50_after(
    group1, group2,group3,
    save_path=Path(cosma_output_path)/"continuum_comp.png"
)

