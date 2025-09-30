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
def Paolo_autofit_global_50_after(group1: QuadrantGroup, group2: QuadrantGroup, use_corrected=False, save_path=None): 
    
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
                    facecolor="red",
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
                    facecolor="green",
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
                
                print('Plotting one autofit subplot...')
                global_autofit1=trail_model_arctic_notch_pushed_plot(x=pixels, 
                                           rho_q=float(3.24651796260324), 
                                           generated_trails=line1.model_full_trail_untrailed,
                                           beta=float(0.5739267086443496), 
                                           w=w, 
                                           A=float(0.1282974150909), 
                                           B=float(0.42615915096883), 
                                           C=float(0.44554343394026996), 
                                           tau_a=float(0.4004902769884765), 
                                           tau_b=float(3.746161545344049), 
                                           tau_c=float(16.16113350890179),
                                           notch=float(51.1)
                                          )
                print('Done!')

                
                ax.plot(pixels, global_autofit1, color='orange', ls='-.', alpha=0.7)
                
                print('Plotting one autofit subplot...')
                global_autofit2=trail_model_exp(x=pixels, 
                                           rho_q=float(0.02613435498378), 
                                           n_e=np.repeat(line2.mean_flux, ut.trail_length), 
                                           n_bg=np.repeat(line2.mean_background, ut.trail_length),
                                          # n_e=line.model_flux, 
                                           #n_bg=line.model_background, 
                                           row=np.repeat(line2.mean_row, ut.trail_length), 
                                           beta=float(0.5739267086443496), 
                                           w=w, 
                                           A=float(0.1282974150909), 
                                           B=float(0.42615915096883), 
                                           C=float(0.44554343394026996), 
                                           tau_a=float(0.4004902769884765), 
                                           tau_b=float(3.746161545344049), 
                                           tau_c=float(16.16113350890179),
                                           notch=float(51.1)
                                          )
                print('Done!')

                
                ax.plot(pixels, global_autofit2, color='blue', ls='-.', alpha=0.7)
                

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
cosma_dataset_path = path.join(path.sep, "cosma", "home", "dphgals", "rjm", "data", "hst", "cte_april_25", "02_2023")
cosma_output_path = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6", "richard_scripts", "output")
workspace_path = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"

dataset_directory=Path(cosma_dataset_path)

dataset = wp.Dataset(dataset_directory)

group1 = dataset.group("ABCD")

cosma_dataset_path2 = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6", "richard_scripts", "2024_july_opt2_notch511_1.0", "02_2023_2024_july_opt2_notch511_1.0")
cosma_output_path2 = path.join(path.sep, "cosma5", "data", "durham", "dc-barr6", "richard_scripts", "output")
workspace_path2 = "/cosma/home/durham/dc-barr6/warm_pixels_workspace/"

dataset_directory2=Path(cosma_dataset_path2)

dataset2 = wp.Dataset(dataset_directory2)

group2 = dataset2.group("ABCD")


# Call the 50 plot function we just defined    
Paolo_autofit_global_50_after(
    group1, group2,
    save_path=Path(cosma_output_path)/"2024_july_opt2_compare_notch511_02_2023.pdf"
)

