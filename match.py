import RF_bucket
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import json
from scipy.constants import c

from rich.progress import track


tau_max = 0.30
n_particles_test = 400000
n_particles_new_sample = 1000000

#optics = pickle.load(open("simulation_input.pkl","rb"))["optics"]
optics = json.load(open("optics.json","r"))

#parabolic example distribution
def distr(tau, tau_max):
    yp = 1 - (xp/tau_max)**2
    yp[abs(tau)>tau_max] = 0

    return yp

plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("tau")
ax1.set_ylabel("ptau")
RF_bucket.plot_separatrix(optics=optics, ax=ax1)


xp = np.linspace(-0.4,0.4,300)
yp = distr(xp, tau_max=tau_max)
yp_original = yp.copy()
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel("tau")
ax2.set_ylabel("density")
ax2.plot(xp,yp, 'b-', zorder=10, label='initial')

N = int(len(xp)/2)
dx=xp[1]-xp[0]

### Transform distribution of tau to distribution of m (m = normalized Hamiltonian)
m_distr_x = []
m_distr_y = []
for ii in track(range(N), description="Transforming distribution..."):
    jj = len(xp) - ii - 1
    tau = xp[jj]
    dens = yp[jj]
    if dens == 0.:
        continue
    print(f"tau= {tau:.3f}, f(tau) = {dens:.3f}")
    dtau = dx
    m0 = RF_bucket.get_m(tau=tau-dtau/2., ptau=0, optics=optics)
    m1 = RF_bucket.get_m(tau=tau+dtau/2., ptau=0, optics=optics)
    dm = m1 - m0
    m = RF_bucket.get_m(tau=tau, ptau=0, optics=optics)
    tau_test, ptau_test = RF_bucket.get_m_shell(m, n_particles=n_particles_test, optics=optics)
    hist, bin_edges = np.histogram(tau_test, bins=len(xp), range=(min(xp)-dx/2., max(xp)+dx/2.))

    hist = (hist + hist[::-1])/2.
    factor = dens/hist[jj]
    yp -= hist*factor
    m_distr_x.append(m)
    m_distr_y.append(np.sum(hist)*factor/dm)

    #ax2.plot(xp, hist*factor, 'ro')
    

m_distr_x.append(0)
m_distr_y.append(0)

m_distr_x = m_distr_x[::-1]
m_distr_y = m_distr_y[::-1]

max_m = max(m_distr_x)
tau_new = []
ptau_new = []
nparts = n_particles_new_sample
counter = 0
max_y = np.max(m_distr_y)
chunk = 20000
### Acceptance-rejection algorithm sampling from distribution of m and a random "angle"
### Note that the said angle exists in another normalized phase space and is only approximately
### relatable to the angle in the tau-ptau space. 
while counter < nparts:
    m = np.random.random(size=chunk)*max_m
    rand_test = np.random.random(size=chunk)*max_y
    yy = np.interp(m, m_distr_x, m_distr_y)
    
    tau, ptau = RF_bucket.get_tau_ptau_from_m_random_theta(m, optics=optics)
    
    mask = rand_test < yy

    tau_new.extend(list(tau[mask]))
    ptau_new.extend(list(ptau[mask]))
    counter += sum(mask)
    print(counter)

hist, bin_edges = np.histogram(tau_new, bins=len(xp), range=(min(xp)-dx/2., max(xp)+dx/2.))
ax2.plot(xp, hist/sum(hist)*sum(yp_original), 'ro', label='sampled')
ax2.legend()

ax1.hist2d(tau_new,ptau_new, bins=100, range=((-0.4,0.4),(-0.001, 0.001)))



plt.show()
