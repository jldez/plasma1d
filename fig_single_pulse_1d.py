import numpy as np
import sdf
from pylab import *
import os
import matplotlib.pyplot as plt
import sys
import scipy.constants as c
import math
np.seterr(divide='ignore')
rcParams['ps.useafm'] = True
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
matplotlib.rcParams.update({'font.size': 14})

########constatnts###########################################
c = 2.99792458*10.0**8.0 									#[m s-1]
hbar = 1.054571726*10.0**-34.0 								#[kg m2 s-1]
me = 9.10938291*10.0**-31.0 								#[kg]
q = 1.602176565*10.0**-19.0 								#[A s]
eps0 = 8.854187817*10.0**-12.0 								#[A2 s4 kg-1 m-3]
mu0 = 1.2566370614e-6
pi = 3.14159265359 											#[-]
kb = 1.38064852e-23 										#[J K-1]
sigma_atom = 3.0e-20 										#[m2]
room_temperature = 300.0 									#[K]
#############################################################

# parameters
F = 2.5 *1e4												#[kg s-2]
tau = 10.0 *1e-15 											#[s]
lambd = 800.0 *1e-9 ; omega = 2.0*pi*c/(lambd)				#nm ; s-1
N = 1
depth=100

# material
m_redu = 0.75*me 											#kg
n0 = 1.45 													# -
bandGap = 9.0*q 											#kg m2 s-2
rho_at = 2.2011171771e+28 									#[m-3]
tau_r = 1.0													#[s]
rho_trap = 5.0e-3 * rho_at
xi = 0.05

# io
path = "data/F"+str(F/1e4)+"_tau"+str(tau*1e15)+"_l"+str(lambd*1e9)+"_N"+str(N)+".npy"
print "loading " + path
data = load("data/F"+str(F/1e4)+"_tau"+str(tau*1e15)+"_l"+str(lambd*1e9)+"_N"+str(N)+".npy").item()
rho, Te, Ex, energy_density,gamma = data["rho"], data["Te"], data["Ex"], data["energy_density"],data["gamma"]
keld, mre = data["contr_keldysh"], data["contr_mre"]

# domain
zmax = 500.0 *1e-9											#nm
tmax = 4.0*tau 		 										#fs

# discretization & pml
steps_per_lambda = 60.0 ; dz = lambd/steps_per_lambda/n0
depth_index = int(depth*1e-9/dz)
Nz = int(np.floor(zmax/dz + 1)) ; z = np.linspace(0,zmax,Nz)
dtmax = dz/c ; dt = 0.4*dtmax
Nt = int(np.floor(tmax/dt+1)) ; t = np.linspace(0,tmax,Nt)[:-1]*1e15
I_0 = F/(pi**0.5*tau) ; E_0 = (2.0*I_0/(n0*eps0*c))**0.5 ; t_0 = tau*2.0*1e15

# figure layout
nrows, ncols = 4, 1
dx, dy = 1.5, 1
figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))
fig, axes = plt.subplots(nrows,ncols,figsize=figsize)

# plots
axes[0].plot(t-t_0,Ex[depth_index,:]/E_0,color="red")
axes[1].semilogy(t-t_0,rho[depth_index,:]/rho_at,color="indigo",alpha=0.75,label=r"$\mathrm{total}$")
axes[1].semilogy(t-t_0,keld[depth_index,:]/rho_at,color="black",ls="-.",label=r"$\mathrm{field}$",lw=1.3)
axes[1].semilogy(t-t_0,mre[depth_index,:]/rho_at,color="black",ls="--",label=r"$\mathrm{col.}$",lw=1.3)
axes[1].legend(loc=4,framealpha=0,labelspacing=0.1,handletextpad=0.2,handleheight=-0.5,markerscale=0.2)
axes[1].set_ylim(ymin=1e-10,ymax=2.0)
axes[2].plot(t-t_0,Te[depth_index,:]*1.381e-23/1.602e-19,color="darkcyan")
axes[2].set_ylim(ymin=0,ymax=15)
axes[3].semilogy(t-t_0,gamma[0,depth_index,:]/1e15,color="darkgreen",lw=1.0)
axes[3].set_ylim(ymin=1e-2,ymax=10)

# ticks stuff
for i in range(nrows):
	axes[i].yaxis.tick_right()
	axes[i].set_xlim(xmin=-20,xmax=20)
	axes[i].locator_params(nbins=5, axis='x')
axes[0].locator_params(nbins=3, axis='y')
axes[2].locator_params(nbins=3, axis='y')
axes[1].set_yticks([1e-9, 1e-6, 1e-3,1])
plt.setp([a.get_xticklabels() for a in axes[0:2]], visible=False)
plt.setp([a.get_yticklabels() for a in axes[:]], visible=False)

# legend fontsize
leg=axes[1].get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize=12) 

# labels
axes[3].set_xlabel(r"$\tau~[\mathrm {fs}]$")
axes[0].set_ylabel(r"$E/E_0$")
axes[1].set_ylabel(r"$\rho/\rho_{\mathrm{sat}}$")
axes[2].set_ylabel(r"$T_e~[\mathrm{eV}]$")
axes[3].set_ylabel(r"$\gamma_{\mathrm{eff}}~[\mathrm{fs}^{-1}]$")
axes[0].text(-19,1.08,r"$F=2.5~\mathrm{J/cm}^2 ~~~~~~~~~~ \tau=10~\mathrm{fs}$")
axes[0].text(-19,0.7,r"$\mathrm{(a)}$")
axes[1].text(-19,0.05,r"$\mathrm{(b)}$")
axes[2].text(-19,13,r"$\mathrm{(c)}$")
axes[3].text(-19,3.5,r"$\mathrm{(d)}$")

# show()
savefig("fig_single_pulse_1d.pdf")