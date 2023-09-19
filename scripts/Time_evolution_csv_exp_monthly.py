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
import math

def date_to_jd(year,month,day):
    """
    Convert a date to Julian Day.
    
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
        4th ed., Duffet-Smith and Zwart, 2011.
    
    Parameters
    ----------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.
        
    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.
    
    day : float
        Day, may contain fractional part.
    
    Returns
    -------
    jd : float
        Julian Day
        
    Examples
    --------
    Convert 6 a.m., February 17, 1985 to Julian Day
    
    >>> date_to_jd(1985,2,17.25)
    2446113.75
    
    """
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
        
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
        
    D = math.trunc(30.6001 * (monthp + 1))
    
    jd = B + C + D + day + 1720994.5
    
    return jd

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
temp_switch_date_since_launch=temp_switch_date-launch_date


# Point to the csv_files directory
csv_path = path.join("csv_files_v4_pushed")

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
success_metric=[]
log_likelihoods=[]
BICs=[]

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
rho_q_exp_upper=[]
rho_q_exp_lower=[]


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
    # Extract success metric values
    successstring=str(data.loc[[17],:])
    successval=successstring.partition("= ")[2]
    success_metric.append(float(successval))
    
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
    temp_BIC=8*np.log(12*50)-2*value
    BICs.append(temp_BIC)
    BICs_squared.append(temp_BIC**2)
    
# Now look for the errors from the uncorrected files 
print('Now locating uncorrected csv files')
files_string_uncorrected=[x for x in files_string if 'corrected' not in x]
files_uncorrected=[]
for stuff in files_string_uncorrected:
    files_uncorrected.append(Path(stuff))

for file in files_uncorrected:
    data = pd.read_csv(f"{file}", header=None)
    # Extract MJD values
    rho_q_exp_string=str(data.loc[[11],:])
    rho_q_exp_val=rho_q_exp_string.partition("= ")[2]
    rho_q_exp.append(float(rho_q_exp_val))
    
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
            if 'TrailModel (N=8)' in long_string:
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
    # Beta
    beta_region_long=find_between(rho_q_region_long, 'beta', 'c' )
    beta_lower_range=float(find_between(beta_region_long, '(', ',' ))
    beta_lower.append(float(betas[file_counter]-beta_lower_range))
    beta_upper_range=float(find_between(beta_region_long, ', ', ')' ))
    beta_upper.append(float(beta_upper_range-betas[file_counter]))
    # a
    a_region_long=find_between(rho_q_region_long, ' a ', ' b ' )
    a_lower_range=float(find_between(a_region_long, '(', ',' ))
    a_lower.append(float(a_vals[file_counter]-a_lower_range))
    a_upper_range=float(find_between(a_region_long, ', ', ')' ))
    a_upper.append(float(a_upper_range-a_vals[file_counter]))
    # b
    b_region_long=find_between(rho_q_region_long, ' b ', 'S' )
    b_lower_range=float(find_between(b_region_long, '(', ',' ))
    b_lower.append(float(b_vals[file_counter]-b_lower_range))
    b_upper_range=float(find_between(b_region_long, ', ', ')' ))
    b_upper.append(float(b_upper_range-b_vals[file_counter]))
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
    # notch
    notch_region_long=find_between(rho_q_region_long, 'notch ', ')' )
    notch_lower_range=float(find_between(notch_region_long, '(', ',' ))
    notch_lower.append(float(notches[file_counter]-notch_lower_range))
    specific_char = ","
    notch_upper_range = float(extract_characters(notch_region_long, specific_char))
    notch_upper.append(float(notch_upper_range-notches[file_counter]))
    file_counter=file_counter+1
    
rho_q_exp_lower_ranges=[]
rho_q_exp_upper_ranges=[]
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
            if 'TrailModelPrint' in long_string:
                info_file=long_string
                
    # Rho q exp before correction
    rho_q_exp_region_long=find_between( info_file, 'Summary (3.0 sigma limits):', 'ummary (1.0 sigma limits):' )
    rho_q_exp_lower_range=float(find_between(rho_q_exp_region_long, '(', ',' ))
    rho_q_exp_lower_ranges.append(rho_q_exp_lower_range)
    rho_q_exp_lower.append(float(rho_q_exp[file_counter]-rho_q_exp_lower_range))
    rho_q_exp_upper_range=float(find_between(rho_q_exp_region_long, ', ', ')' ))
    rho_q_exp_upper_ranges.append(rho_q_exp_upper_range)
    rho_q_exp_upper.append(float(rho_q_exp_upper_range-rho_q_exp[file_counter]))
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
    ax.errorbar(MJDs[i],betas[i], yerr=[[beta_lower[i]], [beta_upper[i]]], color=color,marker="o", linestyle='none')
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
plt.savefig('v4_pushed_plots/Beta(MJD)', bbox_inches="tight")
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
print('')
print('')
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
plt.savefig('v4_pushed_plots/instant_unpushed', bbox_inches="tight")
plt.show()

# relative densities plot
# fit relative densities before and after temp switch date
early_days_list=[]
late_days_list=[]
early_a_list=[]
early_b_list=[]
early_c_list=[]
late_a_list=[]
late_b_list=[]
late_c_list=[]
early_a_err_list=[]
early_b_err_list=[]
early_c_err_list=[]
late_a_err_list=[]
late_b_err_list=[]
late_c_err_list=[]

for i in range(len(ccdgains)):
    if days[i] < temp_switch_date_since_launch:
        early_days_list.append(days[i])
        early_a_list.append(a_vals[i])
        early_b_list.append(b_vals[i])
        early_c_list.append(c_vals[i])
        if a_lower[i] > a_upper[i]:
            early_a_err_list.append(a_lower[i])
        else:
            early_a_err_list.append(a_upper[i])
        if b_lower[i] > b_upper[i]:
            early_b_err_list.append(b_lower[i])
        else:
            early_b_err_list.append(b_upper[i])
    elif days[i] < 1666 or days[i] > 3666:
        late_days_list.append(days[i])
        late_a_list.append(a_vals[i])
        late_b_list.append(b_vals[i])
        late_c_list.append(c_vals[i])
        if a_lower[i] > a_upper[i]:
            late_a_err_list.append(a_lower[i])
        else:
            late_a_err_list.append(a_upper[i])
        if b_lower[i] > b_upper[i]:
            late_b_err_list.append(b_lower[i])
        else:
            late_b_err_list.append(b_upper[i])
            
early_days=np.array(early_days_list)
late_days=np.array(late_days_list)
early_a=np.array(early_a_list)
early_b=np.array(early_b_list)
early_c=np.array(early_c_list)
late_a=np.array(late_a_list)
late_b=np.array(late_b_list)
late_c=np.array(late_c_list)
early_a_err=np.array(early_a_err_list)
early_b_err=np.array(early_b_err_list)
early_c_err=np.array(early_c_err_list)
late_a_err=np.array(late_a_err_list)
late_b_err=np.array(late_b_err_list)
late_c_err=np.array(late_c_err_list)

print('EARLY A FIT RESULTS')
initial_values=np.array([0.0004,0.0446])
deg_freedom = len(early_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, early_days, early_a, 
                                                                  early_a_err))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line_early_a = linear_fit(early_days, [sol0,sol1])

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
print('')
print('')
print('EARLY B FIT RESULTS')
initial_values=np.array([0.0004,0.0446])
deg_freedom = len(early_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, early_days, early_b, 
                                                                  early_b_err))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line_early_b = linear_fit(early_days, [sol0,sol1])

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
print('')
print('')        
print('LATE A FIT RESULTS')
initial_values=np.array([0.0004,0.0446])
deg_freedom = len(late_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, late_days, late_a, 
                                                                  late_a_err))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line_late_a = linear_fit(late_days, [sol0,sol1])

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
print('')
print('')
print('LATE B FIT RESULTS')
initial_values=np.array([0.0004,0.0446])
deg_freedom = len(late_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, late_days, late_b, 
                                                                  late_b_err))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line_late_b = linear_fit(late_days, [sol0,sol1])

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
print('')
print('')
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.errorbar(MJDs[i],a_vals[i],yerr=[[a_lower[i]], [a_upper[i]]],
                color=color,marker="o", linestyle='none')
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.errorbar(MJDs[i],b_vals[i],yerr=[[b_lower[i]], [b_upper[i]]],
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
ax_day.plot(early_days,fit_line_early_a,color="black", linestyle='solid',zorder=15)
ax_day.plot(early_days,fit_line_early_b,color="fuchsia", linestyle='solid',zorder=15)
ax_day.plot(late_days,fit_line_late_a,color="black", linestyle='solid',zorder=15)
ax_day.plot(late_days,fit_line_late_b,color="fuchsia", linestyle='solid',zorder=15)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('v4_pushed_plots/a,b,c(MJD)', bbox_inches="tight")
plt.show()

# tau's plot
# fit line before and after temp change
early_days_list=[]
late_days_list=[]
tau_a_early_list=[]
tau_b_early_list=[]
tau_c_early_list=[]
tau_a_late_list=[]
tau_b_late_list=[]
tau_c_late_list=[]
tau_a_early_err_list=[]
tau_b_early_err_list=[]
tau_c_early_err_list=[]
tau_a_late_err_list=[]
tau_b_late_err_list=[]
tau_c_late_err_list=[]

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    if days[i] < temp_switch_date_since_launch:
        early_days_list.append(days[i])
        tau_a_early_list.append(tau_a_vals[i])
        tau_b_early_list.append(tau_b_vals[i])
        tau_c_early_list.append(tau_c_vals[i])
        if tau_a_lower[i] > tau_a_upper[i]:
            tau_a_early_err_list.append(tau_a_lower[i])
        else:
            tau_a_early_err_list.append(tau_a_upper[i])
        if tau_b_lower[i] > tau_b_upper[i]:
            tau_b_early_err_list.append(tau_b_lower[i])
        else:
            tau_b_early_err_list.append(tau_b_upper[i])
        if tau_c_lower[i] > tau_c_upper[i]:
            tau_c_early_err_list.append(tau_c_lower[i])
        else:
            tau_c_early_err_list.append(tau_c_upper[i])
    elif days[i] < 1666 or days[i] > 3666:
        late_days_list.append(days[i])
        tau_a_late_list.append(tau_a_vals[i])
        tau_b_late_list.append(tau_b_vals[i])
        tau_c_late_list.append(tau_c_vals[i])
        if tau_a_lower[i] > tau_a_upper[i]:
            tau_a_late_err_list.append(tau_a_lower[i])
        else:
            tau_a_late_err_list.append(tau_a_upper[i])
        if tau_b_lower[i] > tau_b_upper[i]:
            tau_b_late_err_list.append(tau_b_lower[i])
        else:
            tau_b_late_err_list.append(tau_b_upper[i])
        if tau_c_lower[i] > tau_c_upper[i]:
            tau_c_late_err_list.append(tau_c_lower[i])
        else:
            tau_c_late_err_list.append(tau_c_upper[i])
    
early_days=np.array(early_days_list)
late_days=np.array(late_days_list) 
tau_a_early=np.array(tau_a_early_list) 
tau_b_early=np.array(tau_b_early_list) 
tau_c_early=np.array(tau_c_early_list) 
tau_a_late=np.array(tau_a_late_list) 
tau_b_late=np.array(tau_b_late_list) 
tau_c_late=np.array(tau_c_late_list) 
tau_a_early_err=np.array(tau_a_early_err_list) 
tau_b_early_err=np.array(tau_b_early_err_list) 
tau_c_early_err=np.array(tau_c_early_err_list) 
tau_a_late_err=np.array(tau_a_late_err_list) 
tau_b_late_err=np.array(tau_b_late_err_list) 
tau_c_late_err=np.array(tau_c_late_err_list)         
        
# Do the linear fit for the tau_a plot and tau_b plot
def linear_fit(x, param_vals):
    #return (param_vals[0]*x**2+param_vals[1]*x+param_vals[2])
    #return (param_vals[0]*x**3+param_vals[1]*x**2+param_vals[2]*x+param_vals[3])
    return (param_vals[0]*x+param_vals[1])

def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, model_params))/y_err)**2)

print('EARLY LINEAR FIT RESULTS FOR TAU_A')
initial_values=np.array([0,0])
deg_freedom = len(early_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, early_days, tau_a_early, 
                                                                  tau_a_early_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_tau_a_early = linear_fit(early_days, [sol0,sol1])

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
print('')
print('')
print('EARLY LINEAR FIT RESULTS FOR TAU_B')
initial_values=np.array([0,0])
deg_freedom = len(early_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, early_days, tau_b_early, 
                                                                  tau_b_early_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_tau_b_early = linear_fit(early_days, [sol0,sol1])

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
print('')
print('')
print('EARLY LINEAR FIT RESULTS FOR TAU_C')
initial_values=np.array([0,0])
deg_freedom = len(early_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, early_days, tau_c_early, 
                                                                  tau_c_early_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_tau_c_early = linear_fit(early_days, [sol0,sol1])

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
print('')
print('')
print('LATE LINEAR FIT RESULTS FOR TAU_A')
initial_values=np.array([0,0])
deg_freedom = len(late_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, late_days, tau_a_late, 
                                                                  tau_a_late_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_tau_a_late = linear_fit( late_days, [sol0,sol1])

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
print('')
print('')
print('LATE LINEAR FIT RESULTS FOR TAU_B')
initial_values=np.array([0,0])
deg_freedom = len(late_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, late_days, tau_b_late, 
                                                                  tau_b_late_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_tau_b_late = linear_fit( late_days, [sol0,sol1])

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
print('')
print('')
print('LATE LINEAR FIT RESULTS FOR TAU_C')
initial_values=np.array([0,0])
deg_freedom = len(late_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_a = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, late_days, tau_c_late, 
                                                                  tau_c_late_err))
print(fit_a.success) 
print(fit_a.message) 
sol0 = fit_a.x[0]
sol1 = fit_a.x[1]
fit_tau_c_late = linear_fit( late_days, [sol0,sol1])

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
print('')
print('')

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
ax_day.plot(early_days,fit_tau_a_early, color='orange', linestyle='solid',zorder=15)
ax_day.plot(early_days,fit_tau_b_early, color='purple', linestyle='solid',zorder=15)
ax_day.plot(early_days,fit_tau_c_early, color='blue', linestyle='solid',zorder=15)
ax_day.plot(late_days,fit_tau_a_late, color='orange', linestyle='solid',zorder=15)
ax_day.plot(late_days,fit_tau_b_late, color='purple', linestyle='solid',zorder=15)
ax_day.plot(late_days,fit_tau_c_late, color='blue', linestyle='solid',zorder=15)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('v4_pushed_plots/tau_a,tau_b,tau_c(MJD)', bbox_inches="tight")
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
plt.savefig('v4_pushed_plots/CCDGAIN(MJD)', bbox_inches="tight")
plt.show()

# success metric plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='purple'
    if ccdgains[i] == 1.0: color='deeppink'
    ax.plot(MJDs[i],success_metric[i], color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Success metric (Rho_q_exp_after/Rho_q_exp_before)', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,ccdgains,marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('v4_pushed_plots/success_metric(MJD)', bbox_inches="tight")
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
plt.savefig('v4_pushed_plots/correction_metrics(MJD)', bbox_inches="tight")
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
print('')
print('')
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
print('')
print('')
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
    ax.errorbar(days[i],notches[i],yerr=[[notch_lower[i]], [notch_upper[i]]],
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
plt.savefig('v4_pushed_plots/Notch(MJD)', bbox_inches="tight")
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
print('RHO_Q ALL DATES LINEAR FIT RESULTS')
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
print('')
print('')




# 2 separate rho_q fits before and after temp switch 
late_rho_days_list=[]
early_rho_days_list=[]
late_rho_list=[]
early_rho_list=[]
late_rho_errors_list=[]
early_rho_errors_list=[]
for i in range(len(ccdgains)):
    if days[i] < temp_switch_date_since_launch:
        early_rho_days_list.append(days[i])
        early_rho_list.append(rho_q_pres[i])
        early_rho_errors_list.append(rho_q_pre_lower[i])
    elif days[i] < 1666 or days[i] > 3666:
        late_rho_days_list.append(days[i])
        late_rho_list.append(rho_q_pres[i])
        late_rho_errors_list.append(rho_q_pre_lower[i])
    
late_rho_days=np.array(late_rho_days_list)
early_rho_days=np.array(early_rho_days_list)
late_rho=np.array(late_rho_list)
early_rho=np.array(early_rho_list)
late_rho_errors=np.array(late_rho_errors_list)
early_rho_errors=np.array(early_rho_errors_list) 

        
# Fit early rho_q values 
print('RHO_Q PRE TEMP SWITCH LINEAR FIT RESULTS')
initial_values=np.array([0.0004,0.0446])
deg_freedom = len(early_rho_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, early_rho_days, early_rho, 
                                                                  early_rho_errors))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line_early = linear_fit(early_rho_days, [sol0,sol1])

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
print('')
print('')
# Fit late rho_q values 
print('RHO_Q POST TEMP SWITCH LINEAR FIT RESULTS')
initial_values=np.array([1,1])
deg_freedom = len(late_rho_days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_fit, late_rho_days, late_rho, 
                                                                  late_rho_errors))
print(fit.success) 
print(fit.message) 
sol0 = fit.x[0]
sol1 = fit.x[1]
fit_line_late = linear_fit(late_rho_days, [sol0,sol1])

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
print('')
print('')
# Find sunspot data 
sun_years=[]
sun_months=[]
sun_days=[]
sunspots=[]
sunspot_JD=[]
sunspot_days_since_launch=[]
sunspot_days_since_launch_fit=[0]
sunspot_fit=[0]
csv_file = 'sunspot_monthly.csv'

# Open the CSV file
with open(csv_file, mode='r') as file:
    # Create a CSV reader with a custom delimiter
    csv_reader = csv.reader(file, delimiter=';')

    # Iterate through each row in the CSV file
    for row in csv_reader:
        # The 'row' variable now contains the data from each row
        if float(row[0]) > 2001:
            sun_years.append(row[0])
            sun_months.append(row[1])
            sun_days.append(15)
            if float(row[3]) > 0:
                sunspots.append(row[3])
            else: sunspots.append(0)

for x in range(len(sunspots)):
    sunspot_JD.append(date_to_jd(float(sun_years[x]),float(sun_months[x]),float(sun_days[x])))
for x in range(len(sunspots)):
    sunspot_days_since_launch.append(sunspot_JD[x]-launch_date_JD)
for x in range(len(sunspots)):
    if sunspot_days_since_launch[x] > 0:
        sunspot_fit.append(sunspots[x])
        sunspot_days_since_launch_fit.append(sunspot_days_since_launch[x])
 
# =============================================================================
# def sunspot_rho_q(date_since_launch, param_vals):
#     sum_sunspots=0
#     rho_q=0
#     for x in range(date_since_launch):
#         sum_sunspots=sum_sunspots+float(sunspot_fit[x+1]) # sum all the sunspots from launch date
#         rho_q=rho_q + float(param_vals[0] + param_vals[1] * ( np.exp(param_vals[2] * (sum_sunspots-param_vals[3])) )) # implement functional form
#         
#     return (sum_sunspots)    
# =============================================================================
def sunspot_rho_q(dates_since_launch, param_vals):
    #sum_sunspots=0
    global dates_since_launchg
    date_since_launchg=dates_since_launch
    rho_q_vals=[]
    for date in dates_since_launch:
        rho_q=0
        for x in range(len(sunspot_days_since_launch_fit)):
            if sunspot_days_since_launch_fit[x]<date:
                rho_q=rho_q + float(param_vals[0] + param_vals[1] * ( np.exp(-param_vals[2] * (float(sunspot_fit[x])-param_vals[3])) )) # implement functional form
        rho_q_vals.append(rho_q)
    return (rho_q_vals)   


def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, model_params))/y_err)**2)

sunspot_error=[]
for i in range(len(rho_q_pres_array)):
    if rho_q_pre_lower[i] > rho_q_pre_upper[i]:
        sunspot_error.append(rho_q_pre_lower[i])
    else:
        sunspot_error.append(rho_q_pre_upper[i])
sunspot_error_array=np.array(sunspot_error)

print('SUNSPOT RHO_Q FIT RESULTS')
initial_values=np.array([0,0,0,0])
deg_freedom = len(days) - initial_values.size
print('DoF = {}'.format(deg_freedom))
fit_sunspot = scipy.optimize.minimize(chi_squared, initial_values, args=(sunspot_rho_q, days, rho_q_pres_array, 
                                                                  sunspot_error_array))
print(fit_sunspot.success) 
print(fit_sunspot.message) 
sol0 = fit_sunspot.x[0]
sol1 = fit_sunspot.x[1]
sol2 = fit_sunspot.x[2]
sol3 = fit_sunspot.x[3]
fit_sunspot_line = sunspot_rho_q(days, [sol0,sol1,sol2,sol3])
fit_sunspot_line_fixed = sunspot_rho_q(days,[0.01,0.005,0.1,15] )

#Show fit results
errs_Hessian = np.sqrt(np.diag(2*fit_sunspot.hess_inv))

zero_err = errs_Hessian[0]
one_err=errs_Hessian[1]
two_err=errs_Hessian[2]
three_err=errs_Hessian[3]


print('minimised chi-squared = {}'.format(fit_sunspot.fun))
chisq_min = fit_sunspot.fun
chisq_reduced = chisq_min/deg_freedom
print('reduced chi^2 = {}'.format(chisq_reduced))
P_value = scipy.stats.chi2.sf(chisq_min, deg_freedom)
print('P(chi^2_min, DoF) = {}'.format(P_value))
print('First coefficient = {} +/- {}'.format(sol0, zero_err))
print('Second coefficient = {} +/- {}'.format(sol1, one_err))
print('Third coefficient = {} +/- {}'.format(sol2, two_err))
print('Fourth coefficient = {} +/- {}'.format(sol3, three_err))
# =============================================================================
# plt.figure(figsize=(8, 6))  # Optional: Set the figure size
# plt.plot(x, sunspot_rho_q(x), color='blue')  # Plot the function
# plt.xlabel('x')  # Label for the x-axis
# plt.ylabel('f(x)')  # Label for the y-axis
# plt.show()  # Display the plot
# =============================================================================
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
# =============================================================================
# for i in range(len(ccdgains)):
#     color2='green'
#     if ccdgains[i] == 1.0: color2='lime'
#     ax.errorbar(days[i],rho_q_exp[i],yerr=[[rho_q_exp_lower[i]], [rho_q_exp_upper[i]]],
#                 color=color2,marker="o", label='exponential pre-correction', linestyle='none', alpha=1) 
# =============================================================================
#ax.plot(days_array, fit_line, linestyle='solid', color='orange')
##ax.plot(late_rho_days, fit_line_late, linestyle='solid', color='black',zorder=10)
##ax.plot(early_rho_days, fit_line_early, linestyle='solid', color='fuchsia',zorder=15)
#ax.plot(days_whole, fit_sunspot_line, linestyle='solid', color='black',zorder=25)
ax.scatter(days, fit_sunspot_line_fixed, color='black',zorder=10)
ax.scatter(days, fit_sunspot_line, color='lime',zorder=10)
ax.set_xlabel("Days since launch", fontsize=12)
ax.set_xlim(-500, max(days)+500)
#ax.set_ylim(-0.02,0.05) # Zoom into post correction rho_q vals
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
plt.xlim(-500, max(days)+500) 
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
plt.savefig('v4_pushed_plots/Rho_q(MJD)', bbox_inches="tight")
plt.show()

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
for i in range(len(ccdgains)):
    color2='green'
    if ccdgains[i] == 1.0: color2='lime'
    ax.errorbar(days[i],rho_q_exp[i],yerr=[[rho_q_exp_lower[i]], [rho_q_exp_upper[i]]],
                color=color2,marker="o", label='exponential pre-correction', linestyle='none', alpha=1) 
ax.plot(days_array, fit_line, linestyle='solid', color='orange')
ax.set_xlabel("Days since launch", fontsize=12)
ax.set_xlim(-500, max(days)+500)
ax.set_ylim(-0.5,0.5) # Zoom into post correction rho_q vals
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
plt.xlim(-500, max(days)+500) 
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
plt.savefig('v4_pushed_plots/Rho_q_post_zoom(MJD)', bbox_inches="tight")
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
print('')
print('')
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