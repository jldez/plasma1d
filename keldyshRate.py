import numpy as np
import math
import scipy.special
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pylab import *
rcParams['ps.useafm'] = True
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
matplotlib.rcParams.update({'font.size': 18})

#nombre de termes pour la somme infinie, constant parce que je suis lache
indiceMax = 10
#domaineLog=[a,b,c,d]. L,intensite laser va de 10^a w/cm^2 a 10^b. Le taux va de 10^c a 10^d en s-1.
domaineLog=[11.0,16.0,0.0,18.0]

########definition des constantes############################
c = 2.99792458*10.0**8.0 									#[m s-1]
hbar = 1.054571726*10.0**-34.0 								#[kg m2 s-1]
me = 9.10938291*10.0**-31.0 								#[kg]
e = 1.602176565*10.0**-19.0 								#[A s]
eps0 = 8.854187817*10.0**-12.0 								#[A2 s4 kg-1 m-3]
pi = 3.14159265359
Navogadro = 6.022*10.0**23.0
IonizationPotentialHydrogen = 13.59844						#eV
energyHartree = 4.35974434*10.0**-18.0 						#kg m2 s-2
omega_au = energyHartree/hbar 								#s-1
#############################################################


########definition des variables independantes###############
#intensite laser
I = np.logspace(domaineLog[0], domaineLog[1], num=500)*10000 #[w m-2]
E = (2.0*I/(c*eps0))**0.5 									#[kg m A-1 s-3]
#longueur donde
lambd = 800.0 												#[nm]
omega = 2.0*pi*c/(lambd*10.0**-9.0) 						#[s-1]
#############################################################


########definition des parametres############################
#Energie dionisation
Ei_eV = 9.0													#[eV]
Ei = Ei_eV*e 												#[kg m2 s-2]
bandGap_eV = 9.0 											#[eV]
bandGap = bandGap_eV*e 										#[kg m2 s-2]

#nombres quantiques
n = (Ei_eV/IonizationPotentialHydrogen)**-0.5
l = 0
m = 0

#dielectriques
mEtoile = 0.75*me 											#kg
densiteMasse = 2.196 										#g/cm3
masseMolaire = 60.08										#g/mol
densiteAtome = densiteMasse/masseMolaire*Navogadro*10.0**6.0 #m-3
#############################################################


########calcul des variables dependantes#####################
quantumDefect = n-(Ei_eV/IonizationPotentialHydrogen)**(-0.5)
nEtoile = n-quantumDefect
lEtoile = l-quantumDefect
E0 = (Ei_eV/IonizationPotentialHydrogen)**1.5*5.14220652*10.0**11.0

def gammaKeldysh(E):
	return omega/(e*E)*sqrt(2.0*me*Ei)

def C(n,l):
	return 2.0**(2.0*n)/(n*scipy.special.gamma(n+l+1.0)*scipy.special.gamma(n-l))

def f(l,m):
	am=abs(m)
	return (2.0*l+1.0)*math.factorial(l+am)/(2.0**am*math.factorial(am)*math.factorial(l-am))

def beta(x):
	return 2.0*x/sqrt(1.0+x**2.0)

def alpha(x):
	return 2.0*(np.arcsinh(x)-beta(x)/2.0)

def g(x):
	return 3.0/(2.0*x)*((1.0+1.0/(2.0*x**2.0))*np.arcsinh(x)-1.0/beta(x))

def integrand(y,x,m):
	return np.exp(y*y-x*x)*(x*x-y*y)**abs(m)

def Phi(x,m):
	integral = quad(integrand,0,x,args=(x,m))
	return integral[0]

def A(omega,gamma):
	nu = Ei/(hbar*omega)*(1.0+1.0/(2.0*gamma**2.0))
	preFacteur=(4.0*gamma**2.0)/(sqrt(3.0*pi)*math.factorial(abs(m))*(1.0+gamma**2.0))
	megaSomme=0.0
	for kappa in range(int(np.ceil(nu)),indiceMax):
		megaSomme+=np.exp(-alpha(gamma)*(kappa-nu))*Phi(sqrt(beta(gamma)*(kappa-nu)),m)
	return preFacteur*megaSomme

def W(omega,E):
	return omega_au*sqrt(6.0/pi)*C(nEtoile,lEtoile)*f(l,m)*\
				Ei_eV/(2.0*IonizationPotentialHydrogen)*A(omega,gammaKeldysh(E))*\
				(2.0*E0/(E*sqrt(1.0+gammaKeldysh(E)**2.0)))**(2.0*n-abs(m)-1.5)*\
				np.exp(-2.0*E0*g(gammaKeldysh(E))/(3.0*E))

def Wtunnel(E):
	return omega_au*sqrt(6.0/pi)*C(nEtoile,lEtoile)*f(l,m)*\
				Ei_eV/(2.0*IonizationPotentialHydrogen)*\
				(2.0*E0/E)**(2.0*nEtoile-abs(m)-1.5)*\
				np.exp(-2.0*E0/(3.0*E))

def Wmpi(omega,E):
	nu0 = Ei/(hbar*omega)
	kappa = np.ceil(nu0)
	return omega_au*(4.0**(2.0*nEtoile))/(pi*sqrt(2.0))*C(nEtoile,lEtoile)*f(l,m)*\
				Ei_eV/(2.0*IonizationPotentialHydrogen)*\
				nu0**(2.0*nEtoile+2.0*kappa-1.5)*np.exp(2.0*kappa-nu0)*\
				Phi(sqrt(2.0*(kappa-nu0)),m)*(E/E0)**(2.0*kappa)

# i,taux,tauxTunnel,tauxMpi=0,E*0.0,E*0.0,E*0.0
# for i in range(I.shape[0]):
# 	taux[i]=W(omega,E[i])
# 	tauxTunnel[i]=Wtunnel(E[i])
# 	tauxMpi[i]=Wmpi(omega,E[i])
# 	print i
#############################################################


########dielectrique condense################################
def gammaCondense(omega,E):
	return omega/(e*E)*sqrt(mEtoile*bandGap)

def grandGamma(gamma):
	return gamma**2.0/(1.0+gamma**2.0)

def grandXi(gamma):
	return 1.0/(1.0+gamma**2.0)

def alphaCondense(gamma):
	return pi*(scipy.special.ellipk(grandGamma(gamma))-scipy.special.ellipe(grandGamma(gamma)))/\
			(scipy.special.ellipe(grandXi(gamma)))

def betaCondense(gamma):
	return (pi**2.0)/(4.0*scipy.special.ellipk(grandXi(gamma))*scipy.special.ellipe(grandXi(gamma)))

def xCondense(omega,gamma):
	return 2.0/pi*bandGap/(hbar*omega)*scipy.special.ellipe(grandXi(gamma))/sqrt(grandGamma(gamma))

def nuCondense(omega,gamma):
	return np.floor(xCondense(omega,gamma)+1.0)-xCondense(omega,gamma)

def indiceMaxAdaptatif(gamma):
	i=1
	if (gamma>1.5):
		i=3
	if (gamma<2.0):
		i=10
	if (gamma<1.5):
		i=20
	if (gamma<0.3):
		i=30
	if (gamma<0.15):
		i=100
	return i

def Q(omega,gamma,x):
	preFacteur = sqrt(pi/(2.0*scipy.special.ellipk(grandXi(gamma))))
	megaSomme = 0
	for n in range(indiceMaxAdaptatif(gamma)):
		megaSomme+=np.exp(-n*alphaCondense(gamma))*\
			Phi(sqrt(betaCondense(gamma)*(n+2.0*nuCondense(omega,gamma))),0)
	return preFacteur*megaSomme

def Wcondense(omega,E):
	return 2.0*omega/(9.0*pi*densiteAtome)*(omega*mEtoile/(hbar*sqrt(grandGamma(gammaCondense(omega,E)))))**1.5*\
				Q(omega,gammaCondense(omega,E),xCondense(omega,gammaCondense(omega,E)))*\
				np.exp(-alphaCondense(gammaCondense(omega,E))*np.floor(xCondense(omega,gammaCondense(omega,E))+1.0))

def WcondenseTunnel(omega,E):
	return 2.0/(9.0*pi**2.0*densiteAtome)*bandGap/hbar*(mEtoile*bandGap/(hbar**2.0))**1.5*\
			(e*hbar*E/(mEtoile**0.5*bandGap**1.5))**2.5*\
			np.exp(-((pi/2.0)*(mEtoile**0.5*bandGap**1.5)/(e*hbar*E)*(1.0+(mEtoile*omega**2.0*bandGap)/(8.0*e**2.0*E**2.0))))

def sigma(omega):
	K = np.ceil(bandGap/(hbar*omega))
	return (2.0*omega/(9.0*pi))*(mEtoile*omega/hbar)**1.5*\
			Phi(sqrt(2.0*(K-bandGap/(hbar*omega))),m)*np.exp(2.0*K)*\
			((e**2.0)/(8.0*mEtoile*omega**2.0*bandGap*eps0*c))**K

def WcondenseMPI(omega,E):
	return (sigma(omega)*(E**2.0*c*eps0/2.0)**(np.ceil(bandGap/(hbar*omega))))/densiteAtome

i,tauxCondense,tauxCondenseTunnel,tauxCondenseMPI=0,E*0.0,E*0.0,E*0.0
for i in range(I.shape[0]):
	tauxCondense[i]=Wcondense(omega,E[i])
	tauxCondenseTunnel[i]=WcondenseTunnel(omega,E[i])
	tauxCondenseMPI[i]=WcondenseMPI(omega,E[i])

# i,tauxCondense,tauxCondenseTunnel,tauxCondenseMPI=0,omega*0.0,omega*0.0,omega*0.0
# for i in range(omega.shape[0]):
# 	tauxCondense[i]=Wcondense(omega[i],E[10])
# plt.plot(lambd,tauxCondense,label="Solid Keldysh")
##############################################################


#######graphiques############################################
# plt.plot(I/10000.0,taux,label="PPT")
# plt.plot(I/10000.0,tauxTunnel,label="Limite tunnel")
# plt.plot(I/10000.0,tauxMpi,label="Limite MPI")
plt.plot(I/10000.0,tauxCondense,label=r"$\nu^{(K)}$",linewidth=1.5,color="darkblue")
plt.plot(I/10000.0,tauxCondenseTunnel,label=r"$\mathrm{Tunnel~limit}$",linewidth=1.5,color="darkgreen")
plt.plot(I/10000.0,tauxCondenseMPI,label=r"$\mathrm{MPI~limit}$",linewidth=1.5,color="darkred")
plt.legend(loc=2,framealpha=0)
#plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$I~[\mathrm{W/cm}^2]$")
# plt.xlabel(r"\lambda~[\mathrm{nm}]")
plt.ylabel(r"$\nu~[\mathrm{s}^{-1}]$")
plt.ylim(10**domaineLog[2],10**domaineLog[3])
#fx,fy,s=0.03,0.4,r"\hspace{-6mm}$E_i=9.0~\mathrm{eV}$\\ $\lambda=800~\mathrm{nm}$"+r"\\ $\rho = 2.654 \cdot 10^{28}~\mathrm{m}^{-3}$"
#plt.text(10**(domaineLog[0]+fx*(domaineLog[1]-domaineLog[0])),10**(domaineLog[2]+fy*(domaineLog[3]-domaineLog[2])),s)
plt.tight_layout()
save("keldyshRate.npy",tauxCondense)
plt.show()
#############################################################


# #######outputEpoch###########################################
# Wepoch = zeros((100000))

# gammaEpoch = linspace(0.01,50.0,100000)
# Eepoch = omega/(e*gammaEpoch)*(mEtoile*bandGap)**0.5

# for i in range(Eepoch.shape[0]):
# 	# print i
# 	Wepoch[i] = Wcondense(omega,Eepoch[i])

# savetxt("ionisationRate.txt",Wepoch)
# #############################################################