from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
from os import path
import sys
import os
import pathlib

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
    
# Generate time evolution plots

# Define some important MJD values
launch_date=2452334.5-2400000.5
repair_dates_1_start=2453912-2400000.5
repair_dates_1_end=2453921-2400000.5
repair_dates_2_start=2454002-2400000.5
repair_dates_2_end=2454018-2400000.5
repair_dates_3_start=2454128-2400000.5
repair_dates_3_end=2454968-2400000.5
temp_switch_date=2453921-2400000.5

# beta plot
plt.figure(figsize=(10, 8))    
plt.plot(MJDs,betas,color="blue",marker="o", linestyle='none') 
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia', label='launch date')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey', label='repair period')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5)
plt.xlabel('MJD', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Beta', fontsize=20)
plt.title('Beta (MJD)', fontsize=20)
plt.legend()
plt.savefig('Beta(MJD)', bbox_inches="tight")
plt.show()

# rho_q plot
plt.figure(figsize=(10, 8))    
plt.plot(MJDs,rho_q_pres,color="blue",marker="o", label='pre-correction', linestyle='none') 
plt.plot(MJDs,rho_q_posts,color="red",marker="o", label='post-correction', linestyle='none') 
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia', label='launch date')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey', label='repair period')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5)
plt.xlabel('MJD', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Rho_q', fontsize=20)
plt.title('Rho_q (MJD)', fontsize=20)
plt.legend()
plt.savefig('Rho_q(MJD)', bbox_inches="tight")
plt.show()

# relative densities plot
plt.figure(figsize=(10, 8))    
plt.plot(MJDs,a_vals,color="red",marker="o", label='a', linestyle='none') 
plt.plot(MJDs,b_vals,color="blue",marker="o", label='b', linestyle='none') 
plt.plot(MJDs,c_vals,color="green",marker="o", label='c', linestyle='none')
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia', label='launch date')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey', label='repair period')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5) 
plt.xlabel('MJD', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Relative Trap Density', fontsize=20)
plt.title('a,b,c (MJD)', fontsize=20)
plt.legend()
plt.savefig('a,b,c(MJD)', bbox_inches="tight")
plt.show()

# tau's plot
plt.figure(figsize=(10, 8))    
plt.plot(MJDs,tau_a_vals,color="red",marker="o", label='tau_a', linestyle='none') 
plt.plot(MJDs,tau_b_vals,color="blue",marker="o", label='tau_b', linestyle='none') 
plt.plot(MJDs,tau_c_vals,color="green",marker="o", label='tau_c', linestyle='none') 
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia', label='launch date')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey', label='repair period')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5)
plt.xlabel('MJD', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Release Timescale', fontsize=20)
plt.title('tau_a,tau_b,tau_c (MJD)', fontsize=20)
plt.legend()
plt.savefig('tau_a,tau_b,tau_c(MJD)', bbox_inches="tight")
plt.show()

# notch plot
plt.figure(figsize=(10, 8))    
plt.plot(MJDs,notches,color="red",marker="o", linestyle='none') 
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia', label='launch date')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey', label='repair period')
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5)
plt.xlabel('MJD', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Notch', fontsize=20)
plt.legend()
plt.title('Notch (MJD)', fontsize=20)
plt.savefig('notch(MJD)', bbox_inches="tight")
plt.show()

# correction metric plots
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
ax.set_title('Correction metrics',fontsize=20)
ax.plot(MJDs,mean_height_reductions,color="red",marker="o", linestyle='none') 
plt.axvline(x=launch_date, ymin=0, ymax=1, color='fuchsia', label='launch date')
plt.axvspan(repair_dates_1_start, repair_dates_1_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_2_start, repair_dates_2_end, alpha=0.5, color='grey', label='repair period')
plt.axvspan(repair_dates_3_start, repair_dates_3_end, alpha=0.5, color='grey', label='repair period')
plt.legend()
plt.axvline(x=temp_switch_date, ymin=0, ymax=1, color='gold', label='temp switch date', alpha=0.5)
ax.set_ylabel("Mean Height Reduction", color="red", fontsize=20)
ax.set_xlabel("MJD", fontsize = 20)
ax2=ax.twinx()
ax2.plot(MJDs,rho_q_reductions,color="blue",marker="o", linestyle='none')  
ax.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylabel("Rho_q Reduction",color="blue",fontsize=20)
plt.savefig('correction_metrics(MJD)', bbox_inches="tight")
plt.show()
