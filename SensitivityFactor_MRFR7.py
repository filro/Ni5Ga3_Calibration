# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:33:52 2022

@author: filro
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import itertools


# CO2, Ar, O2, H2, He, CO

k = np.array([1.114,1.114,1.114,1.114, 1.114])
sigma = np.array([3.521, 3.18, 2.441, 1.021, 8.88e-2])
I_M = np.array([0.76, 0.87, 0.82, 0.98, 1])
M = np.array([44, 40, 32, 2, 4])
T_M = M**-0.355
fM = k*sigma*I_M*T_M

fM = np.append(np.zeros((1,1)), fM)

k_MeOH = k[1]
sigma_MeOH = 4.7
I_M_MeOH = 0.4
T_M_MeOH = T_M
fM_Methanol = k_MeOH*sigma_MeOH*I_M_MeOH*T_M_MeOH[1]

print(fM)

FM = np.array([0, 21.87, 19.18, 9.37, 19.48, 2.8])

# Degree of the fitting polynomial
deg = 1
# Parameters from the fit of the polynomial
p = np.polyfit(fM, FM, deg)
m = p[0]  # Gradient
c = 0  # y-intercept
p[1] = 0

FM_Methanol = m*fM_Methanol
print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')


from scipy import stats

# Number of observations
n = FM.size
# Number of parameters: equal to the degree of the fitted polynomial (ie the
# number of coefficients) plus 1 (ie the number of constants)
m = p.size
# Degrees of freedom (number of observations - number of parameters)
dof = n - m
# Significance level
alpha = 0.05
# We're using a two-sided test
tails = 2
# The percent-point function (aka the quantile function) of the t-distribution
# gives you the critical t-value that must be met in order to get significance
t_critical = stats.t.ppf(1 - (alpha / tails), dof)
# Model the data using the parameters of the fitted straight line
y_model = np.polyval(p, fM)

# Create the linear (1 degree polynomial) model
model = np.poly1d(p)
# Fit the model
y_model = model(fM)

# Mean
y_bar = np.mean(FM)
# Coefficient of determination, R²
R2 = np.sum((y_model - y_bar)**2) / np.sum((FM - y_bar)**2)

print(f'R² = {R2:.2f}')

# Calculate the residuals (the error in the data, according to the model)
resid = FM - y_model
# Chi-squared (estimates the error in data)
chi2 = sum((resid / y_model)**2)
# Reduced chi-squared (measures the goodness-of-fit)
chi2_red = chi2 / dof
# Standard deviation of the error
std_err = np.sqrt(sum(resid**2) / dof)



# Create plot
fig1, ax1 = plt.subplots()
xlim = plt.xlim()
ylim = plt.ylim()
# Line of best fit
plt.plot(np.array(xlim), p[1] + p[0] * np.array(xlim), label=f'Line of Best Fit, R² = {R2:.2f}')
# Fit
x_fitted = np.linspace(xlim[0], xlim[1], 100)
y_fitted = np.polyval(p, x_fitted)
# Confidence interval
ci = t_critical * std_err * np.sqrt(1 / n + (x_fitted - np.mean(fM))**2 / np.sum((fM - np.mean(fM))**2))
plt.fill_between(
    x_fitted, y_fitted + ci, y_fitted - ci, facecolor='#b9cfe7', zorder=0,
    label=r'95% Confidence Interval')
# Prediction Interval
pi = t_critical * std_err * np.sqrt(1 + 1 / n + (x_fitted - np.mean(fM))**2 / np.sum((fM - np.mean(fM))**2))
plt.plot(x_fitted, y_fitted - pi, '--', color='0.5', label=r'95% Prediction Limits')
plt.plot(x_fitted, y_fitted + pi, '--', color='0.5')
# Title and labels
# plt.title('Simple Linear Regression ')
plt.xlabel(r'Calculated Sensitivity Factor $[A \cdot s \slash mol]$')
plt.ylabel(r'Measured Sensitivity Factor  $[A \cdot s \slash mol]$')
# Finished
# plt.legend(fontsize=8)

values = [0, 4, 1, 2, 3, 5]
colors = ListedColormap(['white', 'green', 'red', 'blue', 'dimgray', 'aquamarine'])
Capillary_values = plt.scatter(fM, FM, c=values, cmap=colors, edgecolors='k', label ='Baratron calibrated gases')
Methanol_intersection = plt.scatter(fM_Methanol, FM_Methanol, marker = '^', edgecolor = 'k', color = 'm', label = 'Predicted gases')

plt.xlim(xlim)
plt.ylim(0, ylim[1])
plt.xlim(0,0.9)
plt.ylim(0,30)

Label = ['CO$_2$', 'Ar', 'O$_2$', 'H$_2$', 'He', 'CH$_3$OH']
ax1.annotate(Label[0], (fM[1], FM[1]), xytext=(fM[1]-0.15, FM[1]+0.1), 
    arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

ax1.annotate(Label[1], (fM[2], FM[2]), xytext=(fM[2]-0.05, FM[2]-2.9), 
    arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

ax1.annotate(Label[2], (fM[3], FM[3]), xytext=(fM[3]-0.10, FM[3]-2), 
    arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

ax1.annotate(Label[3], (fM[4], FM[4]), xytext=(fM[4]-0.02, FM[4]+2.9), 
    arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

ax1.annotate(Label[4], (fM[5], FM[5]), xytext=(fM[5]-0.05, FM[5]+2.3), 
    arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

ax1.annotate(Label[5], (fM_Methanol, FM_Methanol), xytext=(fM_Methanol-0.1, FM_Methanol+2.3), 
    arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

t = (f'Methanol F$_M$ = {FM_Methanol:.1f} $\pm$ {std_err:.1f} $A \dot s/mol$')
r = plt.text(0.47, 2, t, ha='left', rotation=0, fontsize = 9, wrap=False, backgroundcolor = 'white')
r.set_bbox(dict(facecolor='green', alpha=0.5, edgecolor='green'))



leg = plt.legend(fontsize = 8, loc='upper left',
          fancybox=False, shadow=False,  ncol=1)
LH = leg.legendHandles
LH[3].set_color('k')
LH[4].set_color('w')
LH[4].set_edgecolor('k')


plt.show()


fig1.savefig('C:/Users/filro/OneDrive - Danmarks Tekniske Universitet/Skrivebord/Data/QMS/QMS_Images/MRFR7_Baratron_Calibration_NoCO.png',
            format='png', dpi=1200, bbox_inches="tight")