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

########constants############################################
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
tau = 10.0 *1e-15 										#[s]
lambd = 800.0 *1e-9 ; omega = 2.0*pi*c/(lambd)				#nm ; s-1
N = 1
gamma_eff = 1.54832251656e+15
alpha_eff = 0.00120633038634

print "calcul F="+str(F/1e4)+", tau="+str(tau*1e15)+", lambda="+str(lambd*1e9)+", N="+str(N) 

#material
m_redu = 0.75*me 											#kg
n0 = 1.45 													# -
bandGap = 9.0*q 											#kg m2 s-2
rho_at = 2.2011171771e+28 									#[m-3]
tau_r = 1.0													#[s]
rho_trap = 0.005 * rho_at
xi = 0.05

# domain
zmax = 500.0 *1e-9											#nm
tmax = 4.0*tau 		 										#fs

# discretization & pml
steps_per_lambda = 60.0 ; dz = lambd/steps_per_lambda/n0
pml_width_per_lambda = 1.0
nb_pml = int(np.floor(lambd*pml_width_per_lambda/dz+1))
zmax = zmax + 2.0*pml_width_per_lambda*lambd
Nz = int(np.floor(zmax/dz + 1)) ; z = np.linspace(0,zmax,Nz)
dtmax = dz/c ; dt = 0.2*dtmax
Nt = int(np.floor(tmax/dt+1)) ; t = np.linspace(0,tmax,Nt)

# conductivity in pml
sigma=zeros((Nz)) ; m=5 #grading order for sigma
sigma_max=(m+1)*8.0/(2*(mu0/(eps0*n0**2.0))**0.5*nb_pml*dz)
sigma[Nz-nb_pml:Nz]=np.linspace(0,1,nb_pml)**m*sigma_max
sigma[0:nb_pml]=np.linspace(1,0,nb_pml)**m*sigma_max
sigma_star = sigma*mu0/(eps0*n0**2.0)

# fields & quantities
Ex,Hy = zeros((Nz)),zeros((Nz))
Jf,Jb,JK = zeros((Nz)),zeros((Nz)),zeros((Nz))
rho,omega_p = zeros((Nz)),zeros((Nz))

# output arrays
Ex_out = zeros((Nz,Nt))
rho_out = zeros((Nz,Nt))
contribution_keldysk = zeros((Nz,Nt))

# Keldysh rate
I_keldysh = np.logspace(11, 16, num=500)*10000
#500pts, echelle log de I=10^11 a 10^16 W/cm2
keldyshRateRaw = load("keldyshRate.npy") 
def keldyshRate(E):
	Ip = 0.5*eps0*c*E**2.0
	# interpolation lin sur echelle log (fonctionne quand meme)
	return np.interp(Ip, I_keldysh, keldyshRateRaw)*1.0

# source
I_0 = F/(pi**0.5*tau) ; E_0 = (2.0*I_0/(n0*eps0*c))**0.5 ; t_0 = tau*2.0
def source(n):
	return E_0*np.exp(-((n*dt-t_0)/tau)**2.0)*np.cos(omega*n*dt)
reflectivity_init = ((1.0-n0)**2.0)/((1.0+n0)**2.0)

# fdtd1d
A=((mu0-0.5*dt*sigma_star)/(mu0+0.5*dt*sigma_star))
B=(dt/dz)/(mu0+0.5*dt*sigma_star)
C=((eps0*n0**2.0-0.5*dt*sigma)/(eps0*n0**2.0+0.5*dt*sigma))
D=(dt/dz)/(eps0*n0**2.0+0.5*dt*sigma)

def update_fields(n):
	Ex[nb_pml] = source(n) * (1.0 - reflectivity)
	Hy[:-1]=A[:-1]*Hy[:-1]-B[:-1]*(Ex[1:]-Ex[:-1])
	Ex[1:-1]=C[1:-1]*Ex[1:-1]-D[1:-1]*(Hy[1:-1]-Hy[:-2]) \
		- dt*(Jf[1:-1]+JK[1:-1])/eps0
	Ex[Nz-1] = Ex[Nz-2]

	Ex_out[:,n] = Ex

	if (n%1000==0):
		print str(n)+"/"+str(Nt)

tau_drude = 1.0/gamma_eff
def update_currents(n):
	Jf[1:-1] = Jf[1:-1]*(1-dt/(2.0*tau_drude)) \
		/(1+dt/(2.0*tau_drude)) \
		+dt*eps0*omega_p[1:-1]**2.0*Ex[1:-1] \
		/(1+dt/(2.0*tau_drude))
	JK[1:-1] = bandGap*keldyshRate(Ex[1:-1])*(rho_at-rho[1:-1]) \
		*Ex[1:-1]/(Ex[1:-1]+1.0)**2.0

nuK = zeros((Nz))
def update_plasma(n):
	#taux d'ionisation Keldysh
	nuK[1:-1] = keldyshRate(Ex[1:-1])

	# saturation
	sat = zeros((Nz))
	sat[1:-1] = (rho_at-rho[1:-1])/rho_at

	# SRE
	def intensity(E):
		return n0*eps0*c/2.0*E**2.0
	contribution_keldysk[1:-1,n] = contribution_keldysk[1:-1,n-1] + dt*rho_at*nuK[1:-1]*sat[1:-1]
	rho[1:-1] = rho[1:-1] + dt*sat[1:-1]*(rho_at*nuK[1:-1] \
		+ alpha_eff*intensity(Ex[1:-1])*rho[1:-1])

	rho_out[:,n] = rho

	omega_p[1:-1] = (q**2.0*rho[1:-1]/(eps0*m_redu))**0.5
	eps_real = n0**2.0 - (omega_p[nb_pml+1]**2.0/omega**2.0) \
		/(1.0+gamma_eff**2.0/omega**2.0)
	eps_imag = (gamma_eff/omega)*(omega_p[nb_pml+1]**2.0/omega**2.0)\
		/(1.0+gamma_eff**2.0/omega**2.0)
	n_real = (((eps_real**2.0+eps_imag**2.0)**0.5 + eps_real)/2.0)**0.5
	n_imag = (((eps_real**2.0+eps_imag**2.0)**0.5 - eps_real)/2.0)**0.5
	return ((1.0-n_real)**2.0 + n_imag**2.0)/((1.0+n_real)**2.0 + n_imag**2.0)

# incubation
if (N>1):
	data = load("data/F"+str(F/1e4)+"_tau"+str(tau*1e15)+"_l"+str(lambd*1e9)+"_N"+str(N-1)+".npy").item()
	rho_previous = data["rho"][:,-1]
	rho_incubation = rho_trap*(1.0-np.exp(-xi*rho_previous/rho_trap))
	rho[nb_pml+1:-nb_pml] = rho_incubation

for n in range(Nt-1)[1:]:
	reflectivity = update_plasma(n)
	update_currents(n)
	update_fields(n)

data_output = {"Ex" : Ex_out[nb_pml+1:-nb_pml,:-1] , "rho" : rho_out[nb_pml+1:-nb_pml,:-1] \
	, "contr_keldysh" : contribution_keldysk[nb_pml+1:-nb_pml,:-1] \
	, "contr_mre" : rho_out[nb_pml+1:-nb_pml,:-1]-contribution_keldysk[nb_pml+1:-nb_pml,:-1]}
path = "data/simpl_F"+str(F/1e4)+"_tau"+str(tau*1e15)+"_l"+str(lambd*1e9)+"_N"+str(N)+".npy"
save(path,data_output)

plot(rho_out[nb_pml+1,:-1])
show()