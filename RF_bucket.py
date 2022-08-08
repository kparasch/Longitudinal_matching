import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import scipy.special
import scipy.optimize

def plot_separatrix(optics=None, ax=None, fmt='r'):
    alfa = optics['alfa']
    V0 = optics['rf_volt_V']
    freq = optics['rf_freq_Hz']
    gamma0 = optics['gamma0']
    beta0 = optics['beta0']
    p0c_eV = optics['p0c_eV']
    length = optics['length']

    eta = alfa - 1./gamma0**2
    #A = 2*np.pi*freq*p0c_eV/c*length/4./V0
    A = 2*V0/(2.*np.pi*freq*p0c_eV/c*length)
    B = 2*np.pi*freq/2./c
    C = eta/2.

    xp = np.linspace(-np.pi/2./B, +np.pi/2./B)
    separatrix = np.sqrt(A/C)*np.cos(B*xp)

    print(f'Max tau  = {np.pi/2./B:.3f} m')
    print(f'Max ptau = {np.sqrt(A/C):.3e}')


    ax.plot(xp, separatrix, fmt)
    ax.plot(xp, -separatrix, fmt)
    return

def get_m(tau=0, ptau=0, optics=None):
    alfa = optics['alfa']
    V0 = optics['rf_volt_V']
    freq = optics['rf_freq_Hz']
    gamma0 = optics['gamma0']
    beta0 = optics['beta0']
    p0c_eV = optics['p0c_eV']
    length = optics['length']

    eta = alfa - 1./gamma0**2
    #A = 2*np.pi*freq*p0c_eV/c*length/4./V0
    A = V0/(2.*np.pi*freq*p0c_eV/c*length)
    B = 2*np.pi*freq/c
    C = eta/2.
    
    m = ( np.sin(B/2.*tau) )**2 + C/2./A*(ptau**2)
    return m

def get_tau_ptau_from_m_random_theta(m, optics):
    alfa = optics['alfa']
    V0 = optics['rf_volt_V']
    freq = optics['rf_freq_Hz']
    gamma0 = optics['gamma0']
    beta0 = optics['beta0']
    p0c_eV = optics['p0c_eV']
    length = optics['length']

    eta = alfa - 1./gamma0**2
    A = V0/(2.*np.pi*freq*p0c_eV/c*length)
    B = 2*np.pi*freq/c
    C = eta/2.

    chunk = len(m)

    theta = np.random.uniform(size=chunk)*2.*np.pi

    K = scipy.special.ellipk(m)
    G = 2.*K/np.pi
    sn, cn, dn, ph = scipy.special.ellipj(G*theta,m)

    tau = 2./B*np.arcsin(np.sqrt(m)*sn)
    ptau = np.sqrt(2*A*m/C)*cn

    return tau, ptau


def get_m_shell(m, n_particles=20000, optics=None):
    alfa = optics['alfa']
    V0 = optics['rf_volt_V']
    freq = optics['rf_freq_Hz']
    gamma0 = optics['gamma0']
    beta0 = optics['beta0']
    p0c_eV = optics['p0c_eV']
    length = optics['length']

    eta = alfa - 1./gamma0**2
    A = V0/(2.*np.pi*freq*p0c_eV/c*length)
    B = 2*np.pi*freq/c
    C = eta/2.

    K = scipy.special.ellipk(m)
    G = 2.*K/np.pi
    theta = np.random.uniform(size=n_particles)*2.*np.pi
    sn, cn, dn, ph = scipy.special.ellipj(G*theta,m)

    tau = 2./B*np.arcsin(np.sqrt(m)*sn)
    ptau = np.sqrt(2*A*m/C)*cn

    return tau, ptau

def get_J_shell(ptau_max=None, m=None, n_particles=20000, optics=None, seed=0):
    np.random.seed(seed+1234567)
    np.random.uniform((1,10000))
    
    alfa = optics['alfa']
    V0 = optics['rf_volt_V']
    freq = optics['rf_freq_Hz']
    gamma0 = optics['gamma0']
    beta0 = optics['beta0']
    p0c_eV = optics['p0c_eV']
    length = optics['length']

    eta = alfa - 1./gamma0**2
    A = V0/(2.*np.pi*freq*p0c_eV/c*length)
    B = 2*np.pi*freq/c
    C = eta/2.

    #max_hamiltonian = A*np.cos(B*max_tau)
    if m is None and ptau_max is not None:
        hamiltonian = A-C*ptau_max**2
        m = (-hamiltonian+A)/(2.*A)
    elif m is not None and ptau_max is None:
        ptau_max = np.sqrt((2*A*m)/C)
    else:
        print('Please provide one and only one of m or ptau_max')
        return None


    
    print(f'ptau_max = {ptau_max:e}')
    print(f'Synchrotron motion m = {m:.3f}')
    K = scipy.special.ellipk(m)
    G = 2.*K/np.pi
    theta = np.random.uniform(size=n_particles)*2.*np.pi
    #action = lambda x: 4*np.sqrt(2*A)/(np.pi*B*np.sqrt(C)) * ( scipy.special.ellipe(x) - ( 1 - x ) * scipy.special.ellipk(x))
    sn, cn, dn, ph = scipy.special.ellipj(G*theta,m)
    #sn= np.sin(theta)
    #cn= np.cos(theta)

    tau = 2./B*np.arcsin(np.sqrt(m)*sn)
    ptau = np.sqrt(2*A*m/C)*cn

    return tau, ptau

def max_action(bunch_length=0.09, optics=None):
    max_tau = np.sqrt(10)*bunch_length

    alfa = optics['alfa']
    V0 = optics['rf_volt_V']
    freq = optics['rf_freq_Hz']
    gamma0 = optics['gamma0']
    beta0 = optics['beta0']
    p0c_eV = optics['p0c_eV']
    length = optics['length']

    eta = alfa - 1./gamma0**2
    A = V0/(2.*np.pi*freq*p0c_eV/c*length)
    B = 2*np.pi*freq/c
    C = eta/2.

    max_hamiltonian = A*np.cos(B*max_tau)

    m = (-max_hamiltonian+A)/(2.*A)
    print(m)
    action = lambda x: 4*np.sqrt(2*A)/(np.pi*B*np.sqrt(C)) * ( scipy.special.ellipe(x) - ( 1 - x ) * scipy.special.ellipk(x))

    sep_action = action(1)
    max_action = action(m)
    for k in range(1,6):
        m0 = scipy.optimize.fsolve( lambda y: (action(y) - k*action(m)/6.), k*m/6.  )
        #m0 = scipy.optimize.fsolve( lambda y: (np.sqrt(action(y)) - k*np.sqrt(action(m))/6.), (k*np.sqrt(m)/6.)**2  )
        ptau = np.sqrt(2*A*m0/C)
        print(k*m/6., m0, action(m0), k*action(m)/6., ptau)

    print(np.sqrt(2*A/C))
