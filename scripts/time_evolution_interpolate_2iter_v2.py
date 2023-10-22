from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
from os import path
import sys
import os
import pathlib
#from astropy.time import Time
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
csv_path = path.join("csv_files_opt8_add")

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
rho_q_exp=[]
log_likelihoods=[]
BICs=[]

# Lists for the errors
rho_q_post_upper=[]
rho_q_post_lower=[]



# Read each csv file
for file in files_corrected:
    data = pd.read_csv(f"{file}", header=None)
    # Extract pre-correction log likelihood
    log_likelihoodstring=str(data.loc[[1],:])
    log_likelihoodval=log_likelihoodstring.partition("= ")[2]
    log_likelihoods.append(float(log_likelihoodval))
    # Extract MJD values
    MJDstring=str(data.loc[[0],:])
    MJDval=MJDstring.partition("= ")[2]
    MJDs.append(float(MJDval))
    # Extract beta values
    betastring=str(data.loc[[2],:])
    betaval=betastring.partition("= ")[2]
    betas.append(float(betaval))
    # Extract rho_q values before correction 
    rho_q_prestring=str(data.loc[[3],:])
    rho_q_preval=rho_q_prestring.partition("= ")[2]
    rho_q_pres.append(float(rho_q_preval))
    # Extract rho_q values after correction
    rho_q_poststring=str(data.loc[[4],:])
    rho_q_postval=rho_q_poststring.partition("= ")[2]
    rho_q_posts.append(float(rho_q_postval))
    # Extract a values
    astring=str(data.loc[[5],:])
    aval=astring.partition("= ")[2]
    a_vals.append(float(aval))
    # Extract b values
    bstring=str(data.loc[[6],:])
    bval=bstring.partition("= ")[2]
    b_vals.append(float(bval))
    # Extract c values
    cstring=str(data.loc[[7],:])
    cval=cstring.partition("= ")[2]
    c_vals.append(float(cval))
    # Extract tau_a values
    tau_astring=str(data.loc[[8],:])
    tau_aval=tau_astring.partition("= ")[2]
    tau_a_vals.append(float(tau_aval))
    # Extract tau_b values
    tau_bstring=str(data.loc[[9],:])
    tau_bval=tau_bstring.partition("= ")[2]
    tau_b_vals.append(float(tau_bval))
    # Extract tau_c values
    tau_cstring=str(data.loc[[10],:])
    tau_cval=tau_cstring.partition("= ")[2]
    tau_c_vals.append(float(tau_cval))
    # Extract notch values
    notchstring=str(data.loc[[11],:])
    notchval=notchstring.partition("= ")[2]
    notches.append(float(notchval))
    # Extract mean height reduction values
    mhrstring=str(data.loc[[13],:])
    mhrval=mhrstring.partition("= ")[2]
    mean_height_reductions.append(float(mhrval))
    # Extract rho_q reduction values
    rqrstring=str(data.loc[[14],:])
    rqrval=rqrstring.partition("= ")[2]
    rho_q_reductions.append(float(rqrval))
    # Extract rho_q reduction values
    ccdstring=str(data.loc[[15],:])
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
    
BICs_squared =[]   
# Calculate BIC values
for value in log_likelihoods:
    temp_BIC=8*np.log(12*50)-2*value # change 8 to 1 to reflect interpolation?
    BICs.append(temp_BIC)
    BICs_squared.append(temp_BIC**2)

    
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
    ax.plot(MJDs[i],betas[i], color=color,marker="o", linestyle='none')
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
plt.savefig('opt8_add_plots/Beta(MJD)', bbox_inches="tight")
plt.show()

# BICS plot
# Do the linear fit for BICs
def linear_fit(x, param_vals):
    #return (param_vals[0]*x**2+param_vals[1]*x+param_vals[2])
    #return (param_vals[0]*x**3+param_vals[1]*x**2+param_vals[2]*x+param_vals[3])
    return (param_vals[0]*x+param_vals[1])

def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, model_params))/y_err)**2)


BICs_array=np.array(BICs)
MJDs_array=np.array(MJDs)
print('LINEAR FIT RESULTS BICs')
initial_values=np.array([0,0])
deg_freedom = len(notches) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_BICs = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, MJDs_array, BICs_array, 
                                                                  np.full(len(BICs_array),0.01) ))
print(fit_BICs.success) 
print(fit_BICs.message) 
sol0 = fit_BICs.x[0]
sol1 = fit_BICs.x[1]
fit_BICs_line = linear_fit(MJDs_array, [sol0,sol1])

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit_BICs.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]


print('minimised chi-squared = {}'.format(fit_BICs.fun))
chisq_min = fit_BICs.fun
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
print('Number of dates:', len(BICs))
print('Mean BIC value:', np.mean(BICs))
print('BIC RMS:', np.sqrt(np.mean(BICs_squared)))
BICs_early=[]
BICs_late=[]
for i in range(len(BICs)):
    if MJDs[i] < 53065: # first 2 years of operations
        BICs_early.append(BICs[i])
    elif MJDs[i] > 59463: # last 2 years of operations
        BICs_late.append(BICs[i])
print('Mean BIC value in the first 2 years:', np.mean(BICs_early))
print('Mean BIC value in the last 2 years:', np.mean(BICs_late))
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='saddlebrown'
    if ccdgains[i] == 1.0: color='chocolate'
    ax.plot(MJDs[i], BICs[i], color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
#ax.set_ylim(0, 1)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
ax.plot(MJDs_array, fit_BICs_line, linestyle='solid', color='orange')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Bayesian Information Criterion', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,betas,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('opt8_add_plots/instant_unpushed', bbox_inches="tight")
plt.show()

# relative densities plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.plot(MJDs[i],a_vals[i],
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.plot(MJDs[i],b_vals[i],
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
plt.savefig('opt8_add_plots/a,b,c(MJD)', bbox_inches="tight")
plt.show()

fit_mjds=[]
fit_tau_a=[]
fit_tau_b=[]
fit_tau_c=[]
fit_tau_a_err=[]
fit_tau_b_err=[]
fit_tau_c_err=[]
# tau's plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    if MJDs[i] > temp_switch_date:
        fit_mjds.append(MJDs[i])
        fit_tau_a.append(tau_a_vals[i])
        fit_tau_b.append(tau_b_vals[i])
        fit_tau_c.append(tau_c_vals[i])
        fit_tau_a_err.append(0.01)
        fit_tau_b_err.append(0.01)
        fit_tau_c_err.append(0.01)
        
        
# Do the linear fit for the tau_a plot and tau_b plot
def linear_fit(x, param_vals):
    #return (param_vals[0]*x**2+param_vals[1]*x+param_vals[2])
    #return (param_vals[0]*x**3+param_vals[1]*x**2+param_vals[2]*x+param_vals[3])
    return (param_vals[0]*x+param_vals[1])

def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, model_params))/y_err)**2)

fit_mjds_array=np.array(fit_mjds)
fit_tau_a_array=np.array(fit_tau_a)
fit_tau_b_array=np.array(fit_tau_b)
fit_tau_c_array=np.array(fit_tau_c)
print('LINEAR FIT RESULTS FOR TAU_A')
initial_values=np.array([0,0])
deg_freedom = len(notches) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, fit_mjds_array, fit_tau_a_array, 
                                                                  fit_tau_a_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_a_line = linear_fit(fit_mjds_array, [sol0,sol1])

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit_a.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]


print('minimised chi-squared = {}'.format(fit_a.fun))
chisq_min = fit_a.fun
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

print('LINEAR FIT RESULTS FOR TAU_B')
initial_values=np.array([0,0])
deg_freedom = len(notches) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_b = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, fit_mjds_array, fit_tau_b_array, 
                                                                  fit_tau_b_err))
print(fit_b.success) 
print(fit_b.message) 
sol0 = fit_b.x[0]
sol1 = fit_b.x[1]
fit_b_line = linear_fit(fit_mjds_array, [sol0,sol1])

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit_b.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]


print('minimised chi-squared = {}'.format(fit_b.fun))
chisq_min = fit_b.fun
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

print('LINEAR FIT RESULTS FOR TAU_C')
initial_values=np.array([0,0])
deg_freedom = len(notches) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_c = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, fit_mjds_array, fit_tau_c_array, 
                                                                  fit_tau_c_err))
print(fit_c.success) 
print(fit_c.message) 
sol0 = fit_c.x[0]
sol1 = fit_c.x[1]
fit_c_line = linear_fit(fit_mjds_array, [sol0,sol1])

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit_b.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]


print('minimised chi-squared = {}'.format(fit_c.fun))
chisq_min = fit_c.fun
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

for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.plot(MJDs[i],tau_a_vals[i],
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.plot(MJDs[i],tau_b_vals[i],
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='green'
    if ccdgains[i] == 1.0: color='lightgreen'
    ax.plot(MJDs[i],tau_c_vals[i],
                color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
ax.plot(fit_mjds_array, fit_a_line, linestyle='solid', color='orange')
ax.plot(fit_mjds_array, fit_b_line, linestyle='solid', color='purple')
ax.plot(fit_mjds_array, fit_c_line, linestyle='solid', color='cadetblue')
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
plt.savefig('opt8_add_plots/tau_a,tau_b,tau_c(MJD)', bbox_inches="tight")
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
plt.savefig('opt8_add_plots/CCDGAIN(MJD)', bbox_inches="tight")
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
plt.savefig('opt8_add_plots/correction_metrics(MJD)', bbox_inches="tight")
plt.show()

# Look for datasets with days > 4000 to find the average beta value 
late_days=[]
late_days_index=[]
betas_to_average=[]
excel_notches=[]
for i in range(len(days)):
    if days[i] > 4000:
        late_days.append(days[i])
        late_days_index.append(i)
        betas_to_average.append(betas[i])
        excel_notches.append(notches[i])
        
avg_beta=np.mean(betas_to_average)
print('mean beta is', avg_beta)
        
# Look for datasets with days > 4000 to find the average best fit values
later_days=[]
later_days_index=[]
a_to_average=[]
b_to_average=[]
c_to_average=[]
tau_a_to_average=[]
tau_b_to_average=[]
tau_c_to_average=[]
notch_to_average=[]
for i in range(len(days)):
    if days[i] > 4000:
        later_days.append(days[i])
        later_days_index.append(i)
        a_to_average.append(a_vals[i])
        b_to_average.append(b_vals[i])
        c_to_average.append(c_vals[i])
        tau_a_to_average.append(tau_a_vals[i])
        tau_b_to_average.append(tau_b_vals[i])
        tau_c_to_average.append(tau_c_vals[i])
        notch_to_average.append(notches[i])
        
avg_a=np.mean(a_to_average)
print('mean a is ',avg_a)
avg_b=np.mean(b_to_average)
print('mean b is ',avg_b)
avg_c=np.mean(c_to_average)
print('mean c is ',avg_c)
avg_tau_a=np.mean(tau_a_to_average)
print('mean tau_a is ',avg_tau_a)
avg_tau_b=np.mean(tau_b_to_average)
print('mean tau_b is ',avg_tau_b)
avg_tau_c=np.mean(tau_c_to_average)
print('mean tau_c is ',avg_tau_c)
avg_notch=np.mean(notch_to_average)
print('mean notch for t>4000 is ',avg_notch)

# Find average notch at early times
notch_early_to_average=[]
for i in range(len(days)):
    if days[i] < 4000:
        notch_early_to_average.append(notches[i])
avg_notch_early=np.mean(notch_early_to_average)
print('mean notch for t<4000 is ',avg_notch_early)
        
# Find average notch at all times
notch_all_to_average=[]
for i in range(len(days)):
    notch_all_to_average.append(notches[i])
avg_notch_all=np.mean(notch_all_to_average)
print('mean notch all t is ',avg_notch_all)

# =============================================================================
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
    ax.plot(days[i],notches[i],
                color=color,marker="o", linestyle='none')
ax.set_xlim(-500, max(days)+500) 
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
plt.savefig('opt8_add_plots/Notch(MJD)', bbox_inches="tight")
plt.show()


rho_fit_days_list = []
rho_fit_vals_list = []
rho_fit_errors_list = []
for i in range(len(ccdgains)):
        rho_fit_days_list.append(days[i])
        rho_fit_vals_list.append(rho_q_posts[i])
        if rho_q_post_lower[i] > rho_q_post_upper[i]:
            rho_fit_errors_list.append(rho_q_post_lower[i])
        else:
            rho_fit_errors_list.append(rho_q_post_upper[i])
            
rho_fit_days=np.array(rho_fit_days_list)
rho_fit_vals=np.array(rho_fit_vals_list)
rho_fit_errors=np.array(rho_fit_errors_list)

def rho_fit(x, param_vals):
    calc_vals=[]
    day_of_temperature_change= temp_switch_date-launch_date
    print(day_of_temperature_change)
    for days in x:
        if days < day_of_temperature_change:
            calc_vals.append(0) #m1 = vals0, c0 = vals1
        elif days < param_vals[5]:
            calc_vals.append(param_vals[4]+param_vals[2]*(days-day_of_temperature_change)) # m2=vals2 c1=vals4
        elif days > param_vals[5]:
            calc_vals.append(param_vals[4]+param_vals[2]*(param_vals[5]-day_of_temperature_change)+param_vals[3]*(days-param_vals[5])) #m3=vals3 t2=vals5
    return (calc_vals)

print('RHO_Q FIT RESULTS')
initial_values=np.array([0,0,0,0,0,4166])
deg_freedom = len(rho_fit_days_list) - initial_values.size
print('DoF = {}'.format(deg_freedom))
bnds = ((0,10) ,(-10, 10),(0,10),(0,10),(0,8000),(0,8000))
fit = scipy.optimize.minimize(chi_squared, initial_values,args=(rho_fit, rho_fit_days, rho_fit_vals, 
                                                                  rho_fit_errors))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
sol2 = fit.x[2]
sol3 = fit.x[3]
sol4 = fit.x[4]
sol5 = fit.x[5]
fit_line_rho = rho_fit(rho_fit_days, [sol0,sol1,sol2,sol3,sol4,sol5])

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]
two_err=errs_Hessian[2]
three_err=errs_Hessian[3]
four_err=errs_Hessian[4]
five_err=errs_Hessian[5]


print('minimised chi-squared = {}'.format(fit.fun))
chisq_min = fit.fun
chisq_reduced = chisq_min/deg_freedom
print('reduced chi^2 = {}'.format(chisq_reduced))
P_value = scipy.stats.chi2.sf(chisq_min, deg_freedom)
print('P(chi^2_min, DoF) = {}'.format(P_value))
print('c_0 = {} +/- {}'.format(sol1, one_err))
print('m1 = {} +/- {}'.format(sol0, zero_err))
print('m2 = {} +/- {}'.format(sol2, two_err))
print('m3 = {} +/- {}'.format(sol3, three_err))
print('t_1 = {} +/- {}'.format(sol4, four_err))
print('t_2 = {} +/- {}'.format(sol5, five_err))
# =============================================================================
# print('c_0 = {} +/- {}'.format(sol1, one_err))
# print('m1 = {} +/- {}'.format(sol0, zero_err))
# print('m2 = {} +/- {}'.format(sol2, two_err))
# print('m3 = {} +/- {}'.format(sol3, three_err))
# print('m4 = {} +/- {}'.format(sol4, four_err))
# =============================================================================
print('')
print('')
# rho_q plot with swapped x-axes
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.plot(days[i],rho_q_pres[i],
                color=color,marker="o",label='pre-correction', linestyle='none')
for i in range(len(ccdgains)):
    color2='red'
    if ccdgains[i] == 1.0: color2='lightcoral'
    ax.errorbar(days[i],rho_q_posts[i],yerr=[[rho_q_post_lower[i]], [rho_q_post_upper[i]]],
                color=color2,marker="o", label='post-correction', linestyle='none', alpha=1) 
#ax.plot(days_array, fit_line, linestyle='solid', color='orange')
ax.set_xlabel("Days since launch", fontsize=12)
#ax.set_xlim(1000,2000)
#ax.set_ylim(-0.2,0.2)
ax.set_xlim(-500, max(days)+500)
#ax.set_xlim(0, max(days)-5700)
#ax.set_ylim(-0.02,0.05) # Zoom into post correction rho_q vals
ax.set_ylabel('Rho_q', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.scatter(rho_fit_days, fit_line_rho, color='black',zorder=25)
ax_MJD = ax.twiny()
ax_MJD.set_xlim(launch_date-500, max(MJDs)+500)
#ax_MJD.set_xlim(launch_date, max(MJDs)-5700)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax_MJD.set_xlabel("MJD", fontsize = 12)
ax_MJD.plot(MJDs,rho_q_pres,color="red",marker="None", linestyle='none') 
ax_MJD.tick_params(axis='both', which='major', labelsize=12)
# =============================================================================
# plt.gca().axes.get_yaxis().set_ticks([])
# =============================================================================
plt.savefig('opt8_add_plots/Rho_q(MJD)', bbox_inches="tight")
plt.show()

# =============================================================================
# # Plot norm residuals
# ax3=fig.add_axes((0,-0.3,1,0.3))
# #norm_residuals = (rho_q_pres_array - fit_line)/rho_q_pre_lower
# plt.xlabel("Days since launch", fontsize=14)
# plt.ylabel("Norm. Residuals", fontsize=14)
# plt.axhline(3,-1000,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.axhline(-3,-1000,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.ylim(-25,25)
# plt.xlim(-500, max(days)+500) 
# plt.scatter(days_array, norm_residuals,zorder=100)
# #ax3.tick_params(axis='both', which='major', labelsize=14)
# # Plot norm residuals histogram
# ax4=fig.add_axes((1,-0.3,0.3,0.3))
# ax4.tick_params(axis='both', which='major', labelsize=14)
# plt.hist(norm_residuals, np.linspace(-25,25,51), orientation = 'horizontal',alpha=1, zorder=100)
# plt.axhline(3,-100,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.axhline(-3,-100,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.ylim(-25,25)
# plt.gca().axes.get_yaxis().set_ticks([])
# plt.savefig('opt8_add_plots/Rho_q(MJD)', bbox_inches="tight")
# plt.show()
# =============================================================================

# rho_q plot with swapped x-axes
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.plot(days[i],rho_q_pres[i],
                color=color,marker="o",label='pre-correction', linestyle='none')
for i in range(len(ccdgains)):
    color2='red'
    if ccdgains[i] == 1.0: color2='lightcoral'
    ax.errorbar(days[i],rho_q_posts[i],yerr=[[rho_q_post_lower[i]], [rho_q_post_upper[i]]],
                color=color2,marker="o", label='post-correction', linestyle='none', alpha=1) 
ax.set_xlabel("Days since launch", fontsize=12)
ax.set_xlim(-500, max(days)+500)
ax.set_ylim(-0.5,0.5) # Zoom into post correction rho_q vals
ax.set_ylabel('Rho_q', fontsize=12)
ax.scatter(rho_fit_days, fit_line_rho, color='black',zorder=25)
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
plt.savefig('opt8_add_plots/Rho_q_post_zoom(MJD)', bbox_inches="tight")
plt.show()

# =============================================================================
# # Plot norm residuals
# ax3=fig.add_axes((0,-0.3,1,0.3))
# norm_residuals = (rho_q_pres_array - fit_line)/rho_q_pre_lower
# plt.xlabel("Days since launch", fontsize=14)
# plt.ylabel("Norm. Residuals", fontsize=14)
# plt.axhline(3,-1000,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.axhline(-3,-1000,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.ylim(-25,25)
# plt.xlim(-500, max(days)+500) 
# plt.scatter(days_array, norm_residuals,zorder=100)
# #ax3.tick_params(axis='both', which='major', labelsize=14)
# # Plot norm residuals histogram
# ax4=fig.add_axes((1,-0.3,0.3,0.3))
# ax4.tick_params(axis='both', which='major', labelsize=14)
# plt.hist(norm_residuals, np.linspace(-25,25,51), orientation = 'horizontal',alpha=1, zorder=100)
# plt.axhline(3,-100,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.axhline(-3,-100,10000, linestyle='dotted', color='black',linewidth=0.5, zorder=1)
# plt.ylim(-25,25)
# plt.gca().axes.get_yaxis().set_ticks([])
# plt.savefig('opt8_add_plots/Rho_q_post_zoom(MJD)', bbox_inches="tight")
# plt.show()
# =============================================================================
                
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

# Now find the error on the mean values
array_tau_a_before=np.array(tau_a_before)
array_tau_b_before=np.array(tau_b_before)
array_tau_c_before=np.array(tau_c_before)
array_tau_a_after=np.array(tau_a_after)
array_tau_b_after=np.array(tau_b_after)
array_tau_c_after=np.array(tau_c_after)

tau_a_uncer_before=np.sqrt(np.mean(array_tau_a_before**2))/np.sqrt(array_tau_a_before.size-1)
tau_b_uncer_before=np.sqrt(np.mean(array_tau_b_before**2))/np.sqrt(array_tau_b_before.size-1)
tau_c_uncer_before=np.sqrt(np.mean(array_tau_c_before**2))/np.sqrt(array_tau_c_before.size-1)
tau_a_uncer_after=np.sqrt(np.mean(array_tau_a_after**2))/np.sqrt(array_tau_a_after.size-1)
tau_b_uncer_after=np.sqrt(np.mean(array_tau_b_after**2))/np.sqrt(array_tau_b_after.size-1)
tau_c_uncer_after=np.sqrt(np.mean(array_tau_c_after**2))/np.sqrt(array_tau_c_after.size-1)

print('tau_a uncertainty on mean before temp switch is',tau_a_uncer_before)
print('tau_b uncertainty on mean before temp switch is',tau_b_uncer_before)
print('tau_c uncertainty on mean before temp switch is',tau_c_uncer_before)
print('tau_a uncertainty on mean after temp switch is',tau_a_uncer_after)
print('tau_b uncertainty on mean after temp switch is',tau_b_uncer_after)
print('tau_c uncertainty on mean after temp switch is',tau_c_uncer_after)