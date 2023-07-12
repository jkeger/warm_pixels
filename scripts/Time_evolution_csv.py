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
csv_path = path.join("csv_files")

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
    rho_q_region_long=find_between( info_file, 'Summary (3.0 sigma limits):', 'Summary (1.0 sigma limits):' )
    rho_q_lower_range=float(find_between(rho_q_region_long, '(', ',' ))
    rho_q_pre_lower.append(float(rho_q_pres[file_counter]-rho_q_lower_range))
    rho_q_upper_range=float(find_between(rho_q_region_long, ', ', ')' ))
    rho_q_pre_upper.append(float(rho_q_upper_range-rho_q_pres[file_counter]))
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
    b_region_long=find_between(rho_q_region_long, ' b ', 'tau_a ' )
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
    tau_c_region_long=find_between(rho_q_region_long, 'tau_c ', 'notch ' )
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
plt.savefig('latest_plots/Beta(MJD)', bbox_inches="tight")
plt.show()

# rho_q plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='blue'
    if ccdgains[i] == 1.0: color='darkturquoise'
    ax.errorbar(MJDs[i],rho_q_pres[i],yerr=[[rho_q_pre_lower[i]], [rho_q_pre_upper[i]]],
                color=color,marker="o",label='pre-correction', linestyle='none')
for i in range(len(ccdgains)):
    color2='red'
    if ccdgains[i] == 1.0: color2='lightcoral'
# =============================================================================
#     ax.errorbar(MJDs[i],rho_q_posts[i],yerr=0,
#                 color=color2,marker="o", label='post-correction', linestyle='none', alpha=1)
# =============================================================================
    ax.errorbar(MJDs[i],rho_q_posts[i],yerr=[[rho_q_post_lower[i]], [rho_q_post_upper[i]]],
                color=color2,marker="o", label='post-correction', linestyle='none', alpha=1) 
ax.set_xlim(launch_date-500, max(MJDs)+500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Rho_q', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,rho_q_pres,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('latest_plots/Rho_q(MJD)', bbox_inches="tight")
plt.show()

# relative densities plot
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
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('latest_plots/a,b,c(MJD)', bbox_inches="tight")
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
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,tau_c_vals,color="red",marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('latest_plots/tau_a,tau_b,tau_c(MJD)', bbox_inches="tight")
plt.show()

# notch plot
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(len(ccdgains)):
    color='red'
    if ccdgains[i] == 1.0: color='lightcoral'
    ax.errorbar(MJDs[i],notches[i],yerr=[[notch_lower[i]], [notch_upper[i]]],
                color=color,marker="o", linestyle='none')
ax.set_xlim(launch_date-500, max(MJDs)+500)
ax.set_ylim(-500, 500)
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', alpha=0.5)
ax.set_ylabel('Notch Depth', fontsize=12)
ax.set_xlabel("MJD", fontsize = 12)
ax_day = ax.twiny()
ax_day.set_xlabel("Days since launch", fontsize=12)
ax_day.set_xlim(-500, max(days)+500)
ax_day.plot(days,notches,marker="None", linestyle='none') 
ax.tick_params(axis='both', which='major', labelsize=12)
ax_day.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('latest_plots/Notch(MJD)', bbox_inches="tight")
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
plt.savefig('latest_plots/CCDGAIN(MJD)', bbox_inches="tight")
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
plt.savefig('latest_plots/correction_metrics(MJD)', bbox_inches="tight")
plt.show()

# Look for datasets with days > 3000 to find the average beta value 
late_days=[]
late_days_index=[]
betas_to_average=[]
for i in range(len(days)):
    if days[i] > 2500 and days[i] < 3500:
        print(i, days[i], mean_height_reductions[i])
        
        