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
from warm_pixels.hst_functions.trail_model_k_fastest import trail_model_arctic_notch_pushed 
from warm_pixels.hst_functions.trail_model_k_fastest import trail_model_arctic_notch_pushed_plot
#from warm_pixels.fit.model import TrailModel
#from warm_pixels.fit.analysis import Analysis
from warm_pixels.model.group import QuadrantGroup
from autoarray.fit.fit_dataset import SimpleFit
import time
from matplotlib.ticker import FormatStrFormatter

from os import path
import sys


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")

cosma_id = int(sys.argv[1])
const_fix = float(sys.argv[2])
rho_fix = 1
dataset_date = str(sys.argv[3])

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
# Define classes 
class Analysis(af.Analysis):
        def __init__(self, x, y, noise, generated_trails):
            self.x = x
            self.y = y
            self.noise = noise
            self.generated_trails = generated_trails
    
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
                generated_trails=self.generated_trails
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
def Paolo_autofit_global_50_after(group1: QuadrantGroup, group2: QuadrantGroup, use_corrected=False, save_path=None, 
                                  save_path_newfig1 = None, save_path_newfig2 = None, save_path_newfig3 = None): 
    
    stacked_lines1 = group1.stacked_lines()
    stacked_lines2 = group2.stacked_lines()

    
    #date = stacked_lines.date How can I get the date value from stacked_lines? 
    
    # Define constants and free variables
    # CCD
    
    w = 84700.0
    # Trap species

   
    
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

    
    line_means=[]
    mean_mean=[]
    
    # Define empty lists for the 2 new plots
    newplot1_trails = []
    newplot1_noise = []
    newplot1_wpflux = []
    newplot1_model = []
    newplot2_row = []
    newplot2_trails = []
    newplot2_noise = []
    newplot2_model = []
    newplot3_row = []
    newplot3_trails = []
    newplot3_noise = []
    newplot3_model = []
    
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
                
             
                trail1 = line1.model_trail  # + line.model_background
                    
                noise1 = np.sqrt(np.square(line1.model_trail_noise)+const_fix**2)  # + line.model_background
                
                

                # Check for negative values
                where_pos1 = np.where(trail1 > 0)[0]
                where_neg1 = np.where(trail1 < 0)[0]
                
                
                trail2 = line2.model_trail  # + line.model_background
                    
                noise2 = np.sqrt(np.square(line2.model_trail_noise)+const_fix**2)  # + line.model_background
                
                

                # Check for negative values
                where_pos2 = np.where(trail2 > 0)[0]
                where_neg2 = np.where(trail2 < 0)[0]
                

                # Don't plot the warm pixel itself
                
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
                    color='black',
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
                    facecolor="None",
                    marker="o",
                    edgecolors='red',
                    alpha=0.7,
                    zorder=-1,
                )
                ax.errorbar(
                    pixels[where_neg1],
                    abs(trail1[where_neg1]),
                    yerr=noise1[where_neg1],
                    color='red',
                    fmt="None",
                    alpha=0.7,
                    zorder=-2,
                )
                ax.scatter(
                    pixels[where_neg2],
                    abs(trail2[where_neg2]),
                    color=c,
                    facecolor="None",
                    marker="o",
                    edgecolors='black',
                    alpha=0.7,
                    zorder=-1,
                )
                ax.errorbar(
                    pixels[where_neg2],
                    abs(trail2[where_neg2]),
                    yerr=noise2[where_neg2],
                    color='black',
                    fmt="None",
                    alpha=0.7,
                    zorder=-2,
                )
                
                print('Plotting one autofit subplot...')
                global_autofit1=trail_model_arctic_notch_pushed_plot(x=pixels, 
                                           rho_q=float(3.57461587388066), 
                                           generated_trails=line1.model_full_trail_untrailed,
                                           beta=float(0.5710477904925793), 
                                           w=w, 
                                           A=float(0.09615396632417), 
                                           B=float(0.19369267857417), 
                                           C=float(0.71015335510166), 
                                           tau_a=float(0.3451079121901047), 
                                           tau_b=float(2.0785361464094914), 
                                           tau_c=float(10.983019685809538),
                                           notch=float(75.67069867645216)
                                          )
                print('Done!')

                
                ax.plot(pixels, global_autofit1, color='red', ls=':', alpha=0.7, zorder=5000, lw=3)
                
                # Extract data for 3 new plots
                if i_row == n_row_bins1 - 1:
                    newplot1_trails.append(trail1)
                    newplot1_noise.append(noise1)
                    newplot1_wpflux.append(line1.mean_flux)
                    newplot1_model.append(global_autofit1)
                
                if i_flux == n_flux_bins1 - 1:
                    newplot2_row.append(line1.mean_row)
                    newplot2_trails.append(trail1)
                    newplot2_noise.append(noise1)
                    newplot2_model.append(global_autofit1)
                    
                if i_flux == n_flux_bins1 - 2:
                    newplot3_row.append(line1.mean_row)
                    newplot3_trails.append(trail1)
                    newplot3_noise.append(noise1)
                    newplot3_model.append(global_autofit1)
                
                print('Plotting one autofit subplot...')
                global_autofit2=trail_model_exp(x=pixels, 
                                           rho_q=float(-0.04764876953446), 
                                           n_e=np.repeat(line2.mean_flux, ut.trail_length), 
                                           n_bg=np.repeat(line2.mean_background, ut.trail_length),
                                          # n_e=line.model_flux, 
                                           #n_bg=line.model_background, 
                                           row=np.repeat(line2.mean_row, ut.trail_length), 
                                           beta=float(0.5710477904925793), 
                                           w=w, 
                                           A=float(0.09615396632417), 
                                           B=float(0.19369267857417), 
                                           C=float(0.71015335510166), 
                                           tau_a=float(0.3451079121901047), 
                                           tau_b=float(2.0785361464094914), 
                                           tau_c=float(10.983019685809538),
                                           notch=float(75.67069867645216)
                                          )
                print('Done!')

                
                ax.plot(pixels, global_autofit2, color='blue', ls=':', alpha=0.7, zorder=5000, lw=3)
                

            ax.set_xlim(0.5, ut.trail_length + 0.5)
            ax.set_xticks(np.arange(2, ut.trail_length + 0.1, 2))
            ax.set_xticks(np.arange(1, ut.trail_length + 0.1, 2), minor=True)
            ax.set_xticklabels([int(x) for x in np.arange(2, ut.trail_length + 0.1, 2)], fontsize=0.1) # change font size of x tick labels and remove decimal places

# =============================================================================
#             for label in ax.get_xticklabels(minor=False):
#                 label.set_fontsize(8)
#             for label in ax.get_xticklabels(minor=True):
#                 label.set_fontsize(8)
# =============================================================================
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
# =============================================================================
#             elif i_flux in [2, n_flux_bins1 - 3]:
#                 ax.set_xlabel("Pixel ($\Delta y$)")
# =============================================================================
            elif i_flux == (n_flux_bins1 // 2)-1: # Only middle column
                ax.set_xlabel("                      Pixel ($\Delta y$)")
# =============================================================================
#             if i_flux != 0:
#                 ax.set_yticklabels([])
#             elif i_row in [1, n_row_bins1 - 2]:
#                 ax.set_ylabel("Number of electrons per pixel $n_\mathrm{e}^\mathrm{trail}$")
# =============================================================================
            if i_flux != 0:
                ax.set_yticklabels([])
            elif i_row == n_row_bins1 // 2:  # Only middle row
                ax.set_ylabel("Number of electrons per pixel $n_\mathrm{e}^\mathrm{trail}$")

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
                        r"WP flux $n_\mathrm{e}^\mathrm{wp}$:",
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

# =============================================================================
#     ax.tick_params(axis='x', which='major', labelsize=0.3) # change font size of x ticks
#     ax.tick_params(axis='x', which='minor', labelsize=0.3) # change font size of x ticks
# =============================================================================
    plt.tight_layout()
    mean_mean.append(np.mean(line_means))
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path.name)
        
    print("Total post correction fit processing time: ", time.time() - start_time, "seconds")
    
    # Now also plot the 1st of the 2 new plots. ######################################################################
    sum_of_trails = []
    sum_of_trails_errs=[]
    sum_of_trail_model=[]
    first_pixels=[]
    first_pixels_errs=[]
    first_pixel_model=[]
    second_pixels=[]
    second_pixels_errs=[]
    second_pixel_model=[]
    
    for trail in newplot1_trails:
        sum_of_trails.append(sum(trail))
        first_pixels.append(trail[0])
        second_pixels.append(trail[1])
    for noise in newplot1_noise:
        first_pixels_errs.append(noise[0])
        second_pixels_errs.append(noise[1])
        noises_quad = 0
        for errors in noise:
            noises_quad+=errors**2
        sum_of_trails_errs.append(noises_quad**0.5)
    for model in newplot1_model:
        sum_of_trail_model.append(sum(model))
        first_pixel_model.append(model[0])
        second_pixel_model.append(model[1])
    
    # Convert all to arrays
    newplot1_trails = np.array(newplot1_trails)
    newplot1_noise = np.array(newplot1_noise)
    newplot1_wpflux = np.array(newplot1_wpflux)
    sum_of_trails = np.array(sum_of_trails)
    sum_of_trails_errs= np.array(sum_of_trails_errs)
    sum_of_trail_model= np.array(sum_of_trail_model)
    first_pixels= np.array(first_pixels)
    first_pixels_errs= np.array(first_pixels_errs)
    first_pixel_model= np.array(first_pixel_model)
    second_pixels= np.array(second_pixels)
    second_pixels_errs= np.array(second_pixels_errs)
    second_pixel_model= np.array(second_pixel_model)
        
     # Plot the models first   
    plt.plot(newplot1_wpflux, sum_of_trail_model, color='black', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
    plt.plot(newplot1_wpflux, first_pixel_model, color='red', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
    plt.plot(newplot1_wpflux, second_pixel_model, color='blue', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
        
    # Check for negative values for sum of trail
    where_sum_pos = np.where(sum_of_trails > 0)[0]
    where_sum_neg = np.where(sum_of_trails < 0)[0]
    
    plt.errorbar(newplot1_wpflux[where_sum_pos], sum_of_trails[where_sum_pos], 
                 yerr = sum_of_trails_errs[where_sum_pos], color='black', capsize=2, alpha=0.7, 
                 label = 'Sum of trail positives')
    
    if len(where_sum_neg) > 0:
        plt.scatter(newplot1_wpflux[where_sum_neg],
                        abs(sum_of_trails[where_sum_neg]),facecolor="None", marker="o", edgecolors='black',
                        alpha=0.7, zorder=-1, label='Sum of trail negatives')
        
        plt.errorbar(newplot1_wpflux[where_sum_neg], abs(sum_of_trails[where_sum_neg]), 
                     yerr = sum_of_trails_errs[where_sum_neg], color='black', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'Sum of trail negatives')    
        
    # Check for negative values for first pixels
    where_firstpixels_pos = np.where(first_pixels > 0)[0]
    where_firstpixels_neg = np.where(first_pixels < 0)[0]
    
    plt.errorbar(newplot1_wpflux[where_firstpixels_pos], first_pixels[where_firstpixels_pos], 
                 yerr = first_pixels_errs[where_firstpixels_pos], color='red', capsize=2, alpha=0.7, 
                 label = 'First pixel in trail positives')
    
    if len(where_firstpixels_neg) > 0:
        plt.scatter(newplot1_wpflux[where_firstpixels_neg],
                        abs(first_pixels[where_firstpixels_neg]),facecolor="None", marker="o", edgecolors='red',
                        alpha=0.7, zorder=-1, label='First pixel in trail negatives')
        
        plt.errorbar(newplot1_wpflux[where_firstpixels_neg], abs(first_pixels[where_firstpixels_neg]), 
                     yerr = first_pixels_errs[where_firstpixels_neg], color='red', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'First pixel in trail negatives')
    
    # Check for negative values for second pixels
    where_secondpixels_pos = np.where(second_pixels > 0)[0]
    where_secondpixels_neg = np.where(second_pixels < 0)[0]
    
    plt.errorbar(newplot1_wpflux[where_secondpixels_pos], second_pixels[where_secondpixels_pos], 
                 yerr = second_pixels_errs[where_secondpixels_pos], color='blue', capsize=2, alpha=0.7, 
                 label = 'second pixel in trail positives')
    
    if len(where_secondpixels_neg) > 0:
        plt.scatter(newplot1_wpflux[where_secondpixels_neg],
                        abs(second_pixels[where_secondpixels_neg]),facecolor="None", marker="o", edgecolors='blue',
                        alpha=0.7, zorder=-1, label='second pixel in trail negatives')
        
        plt.errorbar(newplot1_wpflux[where_secondpixels_neg], abs(second_pixels[where_secondpixels_neg]), 
                     yerr = second_pixels_errs[where_secondpixels_neg], color='blue', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'second pixel in trail negatives')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Brightness of warm pixel $n_\mathrm{e}^\mathrm{wp}$')
    plt.ylabel('Number of electrons per pixel $n_\mathrm{e}^\mathrm{trail}$')
    plt.savefig(save_path_newfig1, dpi=200)
    plt.close()
###############################################################################################################################   

# Now also plot the 2nd of the 2 new plots. ######################################################################
    sum_of_trails = []
    sum_of_trails_errs=[]
    sum_of_trail_model=[]
    first_pixels=[]
    first_pixels_errs=[]
    first_pixel_model=[]
    second_pixels=[]
    second_pixels_errs=[]
    second_pixel_model=[]
    
    for trail in newplot2_trails:
        sum_of_trails.append(sum(trail))
        first_pixels.append(trail[0])
        second_pixels.append(trail[1])
    for noise in newplot2_noise:
        first_pixels_errs.append(noise[0])
        second_pixels_errs.append(noise[1])
        noises_quad = 0
        for errors in noise:
            noises_quad+=errors**2
        sum_of_trails_errs.append(noises_quad**0.5)
    for model in newplot2_model:
        sum_of_trail_model.append(sum(model))
        first_pixel_model.append(model[0])
        second_pixel_model.append(model[1])
    
    # Convert all to arrays
    newplot2_trails = np.array(newplot2_trails)
    newplot2_noise = np.array(newplot2_noise)
    newplot2_row = np.array(newplot2_row)
    sum_of_trails = np.array(sum_of_trails)
    sum_of_trails_errs= np.array(sum_of_trails_errs)
    sum_of_trail_model= np.array(sum_of_trail_model)
    first_pixels= np.array(first_pixels)
    first_pixels_errs= np.array(first_pixels_errs)
    first_pixel_model= np.array(first_pixel_model)
    second_pixels= np.array(second_pixels)
    second_pixels_errs= np.array(second_pixels_errs)
    second_pixel_model= np.array(second_pixel_model)
        
     # Plot the models first   
    plt.plot(newplot2_row, sum_of_trail_model, color='black', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
    plt.plot(newplot2_row, first_pixel_model, color='red', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
    plt.plot(newplot2_row, second_pixel_model, color='blue', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
        
    # Check for negative values for sum of trail
    where_sum_pos = np.where(sum_of_trails > 0)[0]
    where_sum_neg = np.where(sum_of_trails < 0)[0]
    
    plt.errorbar(newplot2_row[where_sum_pos], sum_of_trails[where_sum_pos], 
                 yerr = sum_of_trails_errs[where_sum_pos], color='black', capsize=2, alpha=0.7, 
                 label = 'Sum of trail positives')
    
    if len(where_sum_neg) > 0:
        plt.scatter(newplot2_row[where_sum_neg],
                        abs(sum_of_trails[where_sum_neg]),facecolor="None", marker="o", edgecolors='black',
                        alpha=0.7, zorder=-1, label='Sum of trail negatives')
        
        plt.errorbar(newplot2_row[where_sum_neg], abs(sum_of_trails[where_sum_neg]), 
                     yerr = sum_of_trails_errs[where_sum_neg], color='black', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'Sum of trail negatives')    
        
    # Check for negative values for first pixels
    where_firstpixels_pos = np.where(first_pixels > 0)[0]
    where_firstpixels_neg = np.where(first_pixels < 0)[0]
    
    plt.errorbar(newplot2_row[where_firstpixels_pos], first_pixels[where_firstpixels_pos], 
                 yerr = first_pixels_errs[where_firstpixels_pos], color='red', capsize=2, alpha=0.7, 
                 label = 'First pixel in trail positives')
    
    if len(where_firstpixels_neg) > 0:
        plt.scatter(newplot2_row[where_firstpixels_neg],
                        abs(first_pixels[where_firstpixels_neg]),facecolor="None", marker="o", edgecolors='red',
                        alpha=0.7, zorder=-1, label='First pixel in trail negatives')
        
        plt.errorbar(newplot2_row[where_firstpixels_neg], abs(first_pixels[where_firstpixels_neg]), 
                     yerr = first_pixels_errs[where_firstpixels_neg], color='red', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'First pixel in trail negatives')
    
    # Check for negative values for second pixels
    where_secondpixels_pos = np.where(second_pixels > 0)[0]
    where_secondpixels_neg = np.where(second_pixels < 0)[0]
    
    plt.errorbar(newplot2_row[where_secondpixels_pos], second_pixels[where_secondpixels_pos], 
                 yerr = second_pixels_errs[where_secondpixels_pos], color='blue', capsize=2, alpha=0.7, 
                 label = 'second pixel in trail positives')
    
    if len(where_secondpixels_neg) > 0:
        plt.scatter(newplot2_row[where_secondpixels_neg],
                        abs(second_pixels[where_secondpixels_neg]),facecolor="None", marker="o", edgecolors='blue',
                        alpha=0.7, zorder=-1, label='second pixel in trail negatives')
        
        plt.errorbar(newplot2_row[where_secondpixels_neg], abs(second_pixels[where_secondpixels_neg]), 
                     yerr = second_pixels_errs[where_secondpixels_neg], color='blue', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'second pixel in trail negatives')

    plt.xlabel('Row location of warm pixel')
    plt.ylabel('Number of electrons per pixel $n_\mathrm{e}^\mathrm{trail}$')
    plt.savefig(save_path_newfig2, dpi=200)
    plt.close()
###############################################################################################################################    
# Now also plot the 3rd of the 2 new plots. ######################################################################
    sum_of_trails = []
    sum_of_trails_errs=[]
    sum_of_trail_model=[]
    first_pixels=[]
    first_pixels_errs=[]
    first_pixel_model=[]
    second_pixels=[]
    second_pixels_errs=[]
    second_pixel_model=[]
    
    for trail in newplot3_trails:
        sum_of_trails.append(sum(trail))
        first_pixels.append(trail[0])
        second_pixels.append(trail[1])
    for noise in newplot3_noise:
        first_pixels_errs.append(noise[0])
        second_pixels_errs.append(noise[1])
        noises_quad = 0
        for errors in noise:
            noises_quad+=errors**2
        sum_of_trails_errs.append(noises_quad**0.5)
    for model in newplot3_model:
        sum_of_trail_model.append(sum(model))
        first_pixel_model.append(model[0])
        second_pixel_model.append(model[1])
    
    # Convert all to arrays
    newplot3_trails = np.array(newplot3_trails)
    newplot3_noise = np.array(newplot3_noise)
    newplot3_row = np.array(newplot3_row)
    sum_of_trails = np.array(sum_of_trails)
    sum_of_trails_errs= np.array(sum_of_trails_errs)
    sum_of_trail_model= np.array(sum_of_trail_model)
    first_pixels= np.array(first_pixels)
    first_pixels_errs= np.array(first_pixels_errs)
    first_pixel_model= np.array(first_pixel_model)
    second_pixels= np.array(second_pixels)
    second_pixels_errs= np.array(second_pixels_errs)
    second_pixel_model= np.array(second_pixel_model)
        
     # Plot the models first   
    plt.plot(newplot3_row, sum_of_trail_model, color='black', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
    plt.plot(newplot3_row, first_pixel_model, color='red', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
    plt.plot(newplot3_row, second_pixel_model, color='blue', linestyle=':', alpha=0.7, zorder=5000, linewidth=3)
        
    # Check for negative values for sum of trail
    where_sum_pos = np.where(sum_of_trails > 0)[0]
    where_sum_neg = np.where(sum_of_trails < 0)[0]
    
    plt.errorbar(newplot3_row[where_sum_pos], sum_of_trails[where_sum_pos], 
                 yerr = sum_of_trails_errs[where_sum_pos], color='black', capsize=2, alpha=0.7, 
                 label = 'Sum of trail positives')
    
    if len(where_sum_neg) > 0:
        plt.scatter(newplot3_row[where_sum_neg],
                        abs(sum_of_trails[where_sum_neg]),facecolor="None", marker="o", edgecolors='black',
                        alpha=0.7, zorder=-1, label='Sum of trail negatives')
        
        plt.errorbar(newplot3_row[where_sum_neg], abs(sum_of_trails[where_sum_neg]), 
                     yerr = sum_of_trails_errs[where_sum_neg], color='black', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'Sum of trail negatives')    
        
    # Check for negative values for first pixels
    where_firstpixels_pos = np.where(first_pixels > 0)[0]
    where_firstpixels_neg = np.where(first_pixels < 0)[0]
    
    plt.errorbar(newplot3_row[where_firstpixels_pos], first_pixels[where_firstpixels_pos], 
                 yerr = first_pixels_errs[where_firstpixels_pos], color='red', capsize=2, alpha=0.7, 
                 label = 'First pixel in trail positives')
    
    if len(where_firstpixels_neg) > 0:
        plt.scatter(newplot3_row[where_firstpixels_neg],
                        abs(first_pixels[where_firstpixels_neg]),facecolor="None", marker="o", edgecolors='red',
                        alpha=0.7, zorder=-1, label='First pixel in trail negatives')
        
        plt.errorbar(newplot3_row[where_firstpixels_neg], abs(first_pixels[where_firstpixels_neg]), 
                     yerr = first_pixels_errs[where_firstpixels_neg], color='red', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'First pixel in trail negatives')
    
    # Check for negative values for second pixels
    where_secondpixels_pos = np.where(second_pixels > 0)[0]
    where_secondpixels_neg = np.where(second_pixels < 0)[0]
    
    plt.errorbar(newplot3_row[where_secondpixels_pos], second_pixels[where_secondpixels_pos], 
                 yerr = second_pixels_errs[where_secondpixels_pos], color='blue', capsize=2, alpha=0.7, 
                 label = 'second pixel in trail positives')
    
    if len(where_secondpixels_neg) > 0:
        plt.scatter(newplot3_row[where_secondpixels_neg],
                        abs(second_pixels[where_secondpixels_neg]),facecolor="None", marker="o", edgecolors='blue',
                        alpha=0.7, zorder=-1, label='second pixel in trail negatives')
        
        plt.errorbar(newplot3_row[where_secondpixels_neg], abs(second_pixels[where_secondpixels_neg]), 
                     yerr = second_pixels_errs[where_secondpixels_neg], color='blue', fmt='None', alpha=0.7, zorder=-2, 
                     label = 'second pixel in trail negatives')

    plt.xlabel('Row location of warm pixel')
    plt.ylabel('Number of electrons per pixel $n_\mathrm{e}^\mathrm{trail}$')
    plt.savefig(save_path_newfig3, dpi=200)
    plt.close()
###############################################################################################################################    
    

# Import data to be fitted
cosma_dataset_path = path.join(path.sep, "cosma", "home", "dphgals", "rjm", "data", "hst", "cte_all", "2024_07_LmidB")
cosma_output_path = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6", "richard_scripts", "output")
workspace_path = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"

dataset_directory=Path(cosma_dataset_path)

dataset = wp.Dataset(dataset_directory)

group1 = dataset.group("ABCD")

cosma_dataset_path2 = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6", "richard_scripts", "2024_july_freeparams_1.0", "2024_07_LmidB_2024_july_freeparams_1.0")
cosma_output_path2 = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6", "richard_scripts", "output")
workspace_path2 = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"

dataset_directory2=Path(cosma_dataset_path2)

dataset2 = wp.Dataset(dataset_directory2)

group2 = dataset2.group("ABCD")


# Call the 50 plot function we just defined    
Paolo_autofit_global_50_after(
    group1, group2,
    save_path=Path(cosma_output_path)/"2024_july_opt2_compare_notch511_2024_07_LmidB_fp.pdf",
    save_path_newfig1=Path(cosma_output_path)/"2024_july_opt2_compare_notch511_2024_07_LmidB_newfig1_fp3plt.pdf",
    save_path_newfig2=Path(cosma_output_path)/"2024_july_opt2_compare_notch511_2024_07_LmidB_newfig2_fp3plt.pdf",
    save_path_newfig3=Path(cosma_output_path)/"2024_july_opt2_compare_notch511_2024_07_LmidB_newfig3_fp3plt.pdf"
)

