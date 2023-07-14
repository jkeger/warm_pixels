from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
from os import path
import sys
import os
import pathlib
from astropy.time import Time
import scipy.optimize as scpo
import scipy.stats


def jd_to_dec_yr(dates):
    """Convert one or more Julian dates to decimal-year dates."""
    time = Time(dates, format="jd")
    time.format = "decimalyear"
    return time.value

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
def extract_characters(string, specific_character):
    index = string.find(specific_character)  # Find the index of the specific character
    if index != -1 and index + 1 < len(string):
        extracted_chars = string[index + 1:index + 10]  # Extract the first 9 characters after the specific character
        return extracted_chars
    else:
        return None  # Return None if the specific character is not found or there are no characters after it

# Define some important MJD values
launch_date_JD=2452334.5
launch_date=launch_date_JD-2400000.5
repair_dates_1_start=2453912-2400000.5
repair_dates_1_end=2453921-2400000.5
repair_dates_2_start=2454002-2400000.5
repair_dates_2_end=2454018-2400000.5
repair_dates_3_start=2454128-2400000.5
repair_dates_3_end=2454968-2400000.5
temp_switch_date=2453921-2400000.5


# Point to the csv_files directory
csv_path = path.join("csv_files_tau")

# Find all the csv files
print('Finding csv files')
files_all=list(pathlib.Path(csv_path).glob('*.csv'))
files_string=[]
for stuff in files_all:
    files_string.append(str(stuff))
    
# Find the corrected csv files
print('Now locating corrected csv files')
files_string_corrected=[x for x in files_string if 'corrected' in x]
files_corrected=[]
for stuff in files_string_corrected:
    files_corrected.append(Path(stuff))
    #print(stuff)
    
    
# Define lists for the properties to be plotted as functions of time (MJD)
MJDs=[]
JDs=[]
days=[]
betas=[]
rho_q_pres=[]
rho_q_posts=[]
a_vals=[]
b_vals=[]
c_vals=[]
tau_a_vals=[]
tau_b_vals=[]
tau_c_vals=[]
notches=[]
mean_height_reductions=[]
rho_q_reductions=[]
ccdgains=[]

# Lists for the errors
rho_q_post_upper=[]
rho_q_post_lower=[]
rho_q_pre_upper=[]
rho_q_pre_lower=[]
beta_upper=[]
beta_lower=[]
a_upper=[]
a_lower=[]
b_upper=[]
b_lower=[]
tau_a_upper=[]
tau_a_lower=[]
tau_b_upper=[]
tau_b_lower=[]
tau_c_upper=[]
tau_c_lower=[]
notch_upper=[]
notch_lower=[]


# Read each csv file
for file in files_corrected:
    data = pd.read_csv(f"{file}", header=None)
    # Extract MJD values
    MJDstring=str(data.loc[[0],:])
    MJDval=MJDstring.partition("= ")[2]
    MJDs.append(float(MJDval))
    # Extract beta values
    betastring=str(data.loc[[3],:])
    betaval=betastring.partition("= ")[2]
    betas.append(float(betaval))
    # Extract rho_q values before correction 
    rho_q_prestring=str(data.loc[[4],:])
    rho_q_preval=rho_q_prestring.partition("= ")[2]
    rho_q_pres.append(float(rho_q_preval))
    # Extract rho_q values after correction
    rho_q_poststring=str(data.loc[[5],:])
    rho_q_postval=rho_q_poststring.partition("= ")[2]
    rho_q_posts.append(float(rho_q_postval))
    # Extract a values
    astring=str(data.loc[[6],:])
    aval=astring.partition("= ")[2]
    a_vals.append(float(aval))
    # Extract b values
    bstring=str(data.loc[[7],:])
    bval=bstring.partition("= ")[2]
    b_vals.append(float(bval))
    # Extract c values
    cstring=str(data.loc[[8],:])
    cval=cstring.partition("= ")[2]
    c_vals.append(float(cval))
    # Extract tau_a values
    tau_astring=str(data.loc[[9],:])
    tau_aval=tau_astring.partition("= ")[2]
    tau_a_vals.append(float(tau_aval))
    # Extract tau_b values
    tau_bstring=str(data.loc[[10],:])
    tau_bval=tau_bstring.partition("= ")[2]
    tau_b_vals.append(float(tau_bval))
    # Extract tau_c values
    tau_cstring=str(data.loc[[11],:])
    tau_cval=tau_cstring.partition("= ")[2]
    tau_c_vals.append(float(tau_cval))
    # Extract notch values
    notchstring=str(data.loc[[12],:])
    notchval=notchstring.partition("= ")[2]
    notches.append(float(notchval))
    # Extract mean height reduction values
    mhrstring=str(data.loc[[14],:])
    mhrval=mhrstring.partition("= ")[2]
    mean_height_reductions.append(float(mhrval))
    # Extract rho_q reduction values
    rqrstring=str(data.loc[[15],:])
    rqrval=rqrstring.partition("= ")[2]
    rho_q_reductions.append(float(rqrval))
    # Extract rho_q reduction values
    ccdstring=str(data.loc[[16],:])
    ccdval=ccdstring.partition("= ")[2]
    ccdgains.append(float(ccdval))
    
    # Extract post rho_q error
    with open(file, 'r') as named_file:
        reader = csv.reader(named_file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Access the long string in the desired row
            long_string = row[0]  
            if 'Bayesian Evidence' in long_string:
                info_file=long_string
                
    rho_q_region_long=find_between( info_file, 'Summary (3.0 sigma limits):', 'Summary (1.0 sigma limits):' )
    lower_range=float(find_between(rho_q_region_long, '(', ',' ))
    rho_q_post_lower.append(float(rho_q_postval)-lower_range)
    upper_range=float(find_between(rho_q_region_long, ', ', ')' ))
    rho_q_post_upper.append(upper_range-float(rho_q_postval))
    
# Now look for the errors from the uncorrected files 
print('Now locating uncorrected csv files')
files_string_uncorrected=[x for x in files_string if 'corrected' not in x]
files_uncorrected=[]
for stuff in files_string_uncorrected:
    files_uncorrected.append(Path(stuff))
    
rho_q_lower_ranges=[]
rho_q_upper_ranges=[]
file_counter=0
for file in files_uncorrected:
    with open(file, 'r') as file:
        reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Access the long string in the desired row
            long_string = row[0]  # Assuming the long string is in the third column (0-indexed)
            # Process the long string as needed
            #print("Long string:", long_string)
            if 'Bayesian Evidence' in long_string:
                info_file=long_string
                
    # Rho q before correction
    rho_q_region_long=find_between( info_file, 'Summary (3.0 sigma limits):', 'ummary (1.0 sigma limits):' )
    rho_q_lower_range=float(find_between(rho_q_region_long, '(', ',' ))
    rho_q_lower_ranges.append(rho_q_lower_range)
    rho_q_pre_lower.append(float(rho_q_pres[file_counter]-rho_q_lower_range))
    rho_q_upper_range=float(find_between(rho_q_region_long, ', ', ')' ))
    rho_q_upper_ranges.append(rho_q_upper_range)
    rho_q_pre_upper.append(float(rho_q_upper_range-rho_q_pres[file_counter]))
    #file_counter=file_counter+1
# =============================================================================
#     # Beta
#     beta_region_long=find_between(rho_q_region_long, 'beta', 'c' )
#     beta_lower_range=float(find_between(beta_region_long, '(', ',' ))
#     beta_lower.append(float(betas[file_counter]-beta_lower_range))
#     beta_upper_range=float(find_between(beta_region_long, ', ', ')' ))
#     beta_upper.append(float(beta_upper_range-betas[file_counter]))
# =============================================================================
# =============================================================================
#     # a
#     a_region_long=find_between(rho_q_region_long, ' a ', ' b ' )
#     a_lower_range=float(find_between(a_region_long, '(', ',' ))
#     a_lower.append(float(a_vals[file_counter]-a_lower_range))
#     a_upper_range=float(find_between(a_region_long, ', ', ')' ))
#     a_upper.append(float(a_upper_range-a_vals[file_counter]))
#     # b
#     b_region_long=find_between(rho_q_region_long, ' b ', 'tau_a ' )
#     b_lower_range=float(find_between(b_region_long, '(', ',' ))
#     b_lower.append(float(b_vals[file_counter]-b_lower_range))
#     b_upper_range=float(find_between(b_region_long, ', ', ')' ))
#     b_upper.append(float(b_upper_range-b_vals[file_counter]))
# =============================================================================
    # tau_a
    tau_a_region_long=find_between(rho_q_region_long, 'tau_a ', 'tau_b ' )
    tau_a_lower_range=float(find_between(tau_a_region_long, '(', ',' ))
    tau_a_lower.append(float(tau_a_vals[file_counter]-tau_a_lower_range))
    tau_a_upper_range=float(find_between(tau_a_region_long, ', ', ')' ))
    tau_a_upper.append(float(tau_a_upper_range-tau_a_vals[file_counter]))
    # tau_b
    tau_b_region_long=find_between(rho_q_region_long, 'tau_b ', 'tau_c ' )
    tau_b_lower_range=float(find_between(tau_b_region_long, '(', ',' ))
    tau_b_lower.append(float(tau_b_vals[file_counter]-tau_b_lower_range))
    tau_b_upper_range=float(find_between(tau_b_region_long, ', ', ')' ))
    tau_b_upper.append(float(tau_b_upper_range-tau_b_vals[file_counter]))
    # tau_c
    tau_c_region_long=find_between(rho_q_region_long, 'tau_c ', 'S' )
    tau_c_lower_range=float(find_between(tau_c_region_long, '(', ',' ))
    tau_c_lower.append(float(tau_c_vals[file_counter]-tau_c_lower_range))
    tau_c_upper_range=float(find_between(tau_c_region_long, ', ', ')' ))
    tau_c_upper.append(float(tau_c_upper_range-tau_c_vals[file_counter]))
# =============================================================================
#     # notch
#     notch_region_long=find_between(rho_q_region_long, 'notch ', ')' )
#     notch_lower_range=float(find_between(notch_region_long, '(', ',' ))
#     notch_lower.append(float(notches[file_counter]-notch_lower_range))
#     specific_char = ","
#     notch_upper_range = float(extract_characters(notch_region_long, specific_char))
#     notch_upper.append(float(notch_upper_range-notches[file_counter]))
# =============================================================================
    file_counter=file_counter+1
    
# Convert MJD to JD
for MJDdates in MJDs:
    JDs.append(float(MJDdates+2400000.5))
# Convert JDs to days since launch
for JDdates in JDs:
    days.append(JDdates-launch_date_JD)    
    
# Generate time evolution plots

# beta plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.errorbar(MJDs[i],betas[i], yerr=0, color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
#ax.set_ylim(0, 1)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Beta', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,betas,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('tau_plots/Beta(MJD)', bbox_inches="tight")
plt.show()


# relative densities plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.errorbar(MJDs[i],a_vals[i],yerr=0,
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.errorbar(MJDs[i],b_vals[i],yerr=0,
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='green'
    if ccdgains[i] == 1.0: color='lightgreen'
    ax.plot(MJDs[i],c_vals[i], color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Relative Trap Density', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,c_vals,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('tau_plots/a,b,c(MJD)', bbox_inches="tight")
plt.show()

# tau's plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.errorbar(MJDs[i],tau_a_vals[i], yerr=[[tau_a_lower[i]], [tau_a_upper[i]]],
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.errorbar(MJDs[i],tau_b_vals[i],yerr=[[tau_b_lower[i]], [tau_b_upper[i]]],
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='green'
    if ccdgains[i] == 1.0: color='lightgreen'
    ax.errorbar(MJDs[i],tau_c_vals[i],yerr=[[tau_c_lower[i]], [tau_c_upper[i]]],
                color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Release Timescale', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
#ax.set_ylim(0,5) #Zoom into the tau_a values
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,tau_c_vals,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('tau_plots/tau_a,tau_b,tau_c(MJD)', bbox_inches="tight")
plt.show()

# ccdgain plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.plot(MJDs[i],ccdgains[i], color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('CCD Gain', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,ccdgains,marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('tau_plots/CCDGAIN(MJD)', bbox_inches="tight")
plt.show()

# correction metric plots
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
#ax.set_title('Correction metrics',fontsize=20)
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.plot(MJDs[i],mean_height_reductions[i], color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5)
ax.set_ylabel("Mean Height Reduction", color="red", fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax2=ax.twinx()
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax2.plot(MJDs[i],rho_q_reductions[i], color=color,marker="o", linestyle='none')
ax_day.plot(days,mean_height_reductions,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_ylabel("Rho_q Reduction",color="blue",fontsize=12)
plt.savefig('tau_plots/correction_metrics(MJD)', bbox_inches="tight")
plt.show()

# =============================================================================
# # Look for datasets with days > 3000 to find the average beta value 
# late_days=[]
# late_days_index=[]
# betas_to_average=[]
# excel_notches=[]
# for i in range(len(days)):
#     if days[i] > 3000:
#         late_days.append(days[i])
#         late_days_index.append(i)
#         betas_to_average.append(betas[i])
#         excel_notches.append(notches[i])
#         
# avg_beta=np.mean(betas_to_average)
# print('mean beta is', avg_beta)
#         
# # Look for datasets with days > 4000 to find the average best fit values
# later_days=[]
# later_days_index=[]
# a_to_average=[]
# b_to_average=[]
# c_to_average=[]
# tau_a_to_average=[]
# tau_b_to_average=[]
# tau_c_to_average=[]
# notch_to_average=[]
# for i in range(len(days)):
#     if days[i] > 4000:
#         later_days.append(days[i])
#         later_days_index.append(i)
#         a_to_average.append(a_vals[i])
#         b_to_average.append(b_vals[i])
#         c_to_average.append(c_vals[i])
#         tau_a_to_average.append(tau_a_vals[i])
#         tau_b_to_average.append(tau_b_vals[i])
#         tau_c_to_average.append(tau_c_vals[i])
#         notch_to_average.append(notches[i])
#         
# avg_a=np.mean(a_to_average)
# print('mean a is ',avg_a)
# avg_b=np.mean(b_to_average)
# print('mean b is ',avg_b)
# avg_c=np.mean(c_to_average)
# print('mean c is ',avg_c)
# avg_tau_a=np.mean(tau_a_to_average)
# print('mean tau_a is ',avg_tau_a)
# avg_tau_b=np.mean(tau_b_to_average)
# print('mean tau_b is ',avg_tau_b)
# avg_tau_c=np.mean(tau_c_to_average)
# print('mean tau_c is ',avg_tau_c)
# avg_notch=np.mean(notch_to_average)
# print('mean notch for t>4000 is ',avg_notch)
# 
# # Find average notch at early times
# notch_early_to_average=[]
# for i in range(len(days)):
#     if days[i] < 4000:
#         notch_early_to_average.append(notches[i])
# avg_notch_early=np.mean(notch_early_to_average)
# print('mean notch for t<4000 is ',avg_notch_early)
#         
# # Find average notch at all times
# notch_all_to_average=[]
# for i in range(len(days)):
#     notch_all_to_average.append(notches[i])
# avg_notch_all=np.mean(notch_all_to_average)
# print('mean notch all t is ',avg_notch_all)
# 
# # Do the linear fit for the notch plot
# def linear_fit(x, param_vals):
#     #return (param_vals[0]*x**2+param_vals[1]*x+param_vals[2])
#     #return (param_vals[0]*x**3+param_vals[1]*x**2+param_vals[2]*x+param_vals[3])
#     return (param_vals[0]*x+param_vals[1]*5)
# 
# def chi_squared(model_params, model, x_data, y_data, y_err):
#     return np.sum(((y_data - model(x_data, model_params))/y_err)**2)
# 
# days_array=np.array(days)
# notches_array=np.array(notches)
# print('LINEAR FIT RESULTS')
# initial_values=np.array([0,1])
# deg_freedom = len(notches) - initial_values.size
# print('DoF = {}'.format(deg_freedom))
# fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, days_array, notches_array, 
#                                                                  notch_lower))
# print(fit.success) 
# print(fit.message) 
# sol0 = fit.x[0]
# sol1 = fit.x[1]
# fit_line = linear_fit(days_array, [sol0,sol1])
# 
# #Show fit results
# errs_Hessian = np.sqrt(np.diag(2*fit.hess_inv))
# 
# zero_err = errs_Hessian[0]
# one_err=errs_Hessian[1]
# 
# 
# print('minimised chi-squared = {}'.format(fit.fun))
# chisq_min = fit.fun
# chisq_reduced = chisq_min/deg_freedom
# print('reduced chi^2 = {}'.format(chisq_reduced))
# P_value = scipy.stats.chi2.sf(chisq_min, deg_freedom)
# print('P(chi^2_min, DoF) = {}'.format(P_value))
# print('First coefficient = {} +/- {}'.format(sol0, zero_err))
# print('Second coefficient = {} +/- {}'.format(sol1, one_err))
# print('Model Equation: {}x+{}'.format(sol0,sol1))
# linear_coef1=sol0
# linear_coef1_err=zero_err
# linear_coef2=sol1
# linear_coef2_err=one_err
# =============================================================================

# notch plot with swapped x-axes
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
#ax.plot(days_array, fit_line, linestyle='solid')
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.errorbar(days[i],notches[i],yerr=0,
                color=color,marker="o", linestyle='none')
ax.set_xlim(-500, max(days)+800) 
ax.set_ylabel('Notch Depth', fontsize=12)
ax.set_xlabel("Days since launch", fontsize = 12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylim(-500, 500)
ax_MJD = ax.twiny()
ax_MJD.set_xlim(launch_date-500, max(MJDs)+500)
ax_MJD.plot(MJDs,notches,marker="None", linestyle='none')
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax_MJD.set_xlabel("MJD", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('tau_plots/Notch(MJD)', bbox_inches="tight")
plt.show()


# Do the linear fit for the rho plot
def linear_fit(x, param_vals):
    #return (param_vals[0]*x**2+param_vals[1]*x+param_vals[2])
    #return (param_vals[0]*x**3+param_vals[1]*x**2+param_vals[2]*x+param_vals[3])
    return (param_vals[0]*x+param_vals[1])

def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, model_params))/y_err)**2)

days_array=np.array(days)
rho_q_pres_array=np.array(rho_q_pres)
print('LINEAR FIT RESULTS')
initial_values=np.array([0.0004,0.0446])
deg_freedom = len(notches) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, days_array, rho_q_pres_array, 
                                                                  rho_q_pre_lower))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line = linear_fit(days_array, [sol0,sol1])

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]


print('minimised chi-squared = {}'.format(fit.fun))
chisq_min = fit.fun
chisq_reduced = chisq_min/deg_freedom
print('reduced chi^2 = {}'.format(chisq_reduced))
P_value = scipy.stats.chi2.sf(chisq_min, deg_freedom)
print('P(chi^2_min, DoF) = {}'.format(P_value))
print('First coefficient = {} +/- {}'.format(sol0, zero_err))
print('Second coefficient = {} +/- {}'.format(sol1, one_err))
print('Model Equation: {}x+{}'.format(sol0,sol1))
linear_coef1=sol0
linear_coef1_err=zero_err
linear_coef2=sol1
linear_coef2_err=one_err

# rho_q plot with swapped x-axes
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.errorbar(days[i],rho_q_pres[i],yerr=[[rho_q_pre_lower[i]], [rho_q_pre_upper[i]]],
                color=color,marker="o",label='pre-correction', linestyle='none')
for i in range(len(ccdgains)):
    color2='red'
    if ccdgains[i] == 1.0: color2='lightcoral'
    ax.errorbar(days[i],rho_q_posts[i],yerr=[[rho_q_post_lower[i]], [rho_q_post_upper[i]]],
                color=color2,marker="o", label='post-correction', linestyle='none', alpha=1) 
ax.plot(days_array, fit_line, linestyle='solid', color='orange')
ax.set_xlabel("Days since launch", fontsize=12)
ax.set_xlim(-500, max(days)+500)
ax.set_ylim(-0.02,0.05) # Zoom into post correction rho_q vals
ax.set_ylabel('Rho_q', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_MJD = ax.twiny()
ax_MJD.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax_MJD.set_xlabel("MJD", fontsize = 12)
ax_MJD.plot(MJDs,rho_q_pres,color="red",marker="None", linestyle='none') 
ax_MJD.tick_params(axis='both', which='major', labelsize=12)

# Plot norm residuals
ax3=fig.add_axes((0,-0.3,1,0.3))
norm_residuals = (rho_q_pres_array - fit_line)/rho_q_pre_lower
plt.xlabel("Days since launch", fontsize=14)
plt.ylabel("Norm. Residuals", fontsize=14)
plt.axhline(3,-1000,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
plt.axhline(-3,-1000,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
plt.ylim(-25,25)
plt.xlim(-500, max(days)+800) 
plt.scatter(days_array, norm_residuals,zorder=100)
#ax3.tick_params(axis='both', which='major', labelsize=14)
# Plot norm residuals histogram
ax4=fig.add_axes((1,-0.3,0.3,0.3))
ax4.tick_params(axis='both', which='major', labelsize=14)
plt.hist(norm_residuals, np.linspace(-25,25,51), orientation = 'horizontal',alpha=1, zorder=100)
plt.axhline(3,-100,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
plt.axhline(-3,-100,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
plt.ylim(-25,25)
plt.gca().axes.get_yaxis().set_ticks([])
plt.savefig('tau_plots/Rho_q(MJD)', bbox_inches="tight")
plt.show()
                
# Find the mean value of the taus before and after temp switch date
#Look for datasets with days > 3000 to find the average beta value 
switch_day=2453921-launch_date_JD
before_switch_days=[]
after_switch_days=[]
tau_a_before=[]
tau_b_before=[]
tau_c_before=[]
tau_a_after=[]
tau_b_after=[]
tau_c_after=[]
for i in range(len(days)):
    if days[i] < switch_day:
        before_switch_days.append(days[i])
        tau_a_before.append(tau_a_vals[i])
        tau_b_before.append(tau_b_vals[i])
        tau_c_before.append(tau_c_vals[i])
    elif days[i] >= switch_day:
        after_switch_days.append(days[i])
        tau_a_after.append(tau_a_vals[i])
        tau_b_after.append(tau_b_vals[i])
        tau_c_after.append(tau_c_vals[i])
        
avg_t_a_before=np.mean(tau_a_before)
avg_t_b_before=np.mean(tau_b_before)
avg_t_c_before=np.mean(tau_c_before)
avg_t_a_after=np.mean(tau_a_after)
avg_t_b_after=np.mean(tau_b_after)
avg_t_c_after=np.mean(tau_c_after)

print('temp switch day is', switch_day)
print('mean tau_a before temp switch is', avg_t_a_before)
print('mean tau_b before temp switch is', avg_t_b_before)
print('mean tau_c before temp switch is', avg_t_c_before)
print('mean tau_a after temp switch is', avg_t_a_after)
print('mean tau_b after temp switch is', avg_t_b_after)
print('mean tau_c after temp switch is', avg_t_c_after)