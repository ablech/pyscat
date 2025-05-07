import numpy as np
from scipy.special import riccati_jn, riccati_yn, spherical_jn, spherical_yn, hankel1, hankel2


class Collision(object):
    
    def __init__(self, mass, V, r):
        """Simulate a collision using the coupled-channels equations."""
        self.mass = mass
        self.V = V
        self.r = r
        
    def solve(self, energies, l_vals, method):
        """Solve the coupled-channels equations using the renormalized Numerov
        or the log-derivative method.
        
        Parameters
        ----------
        energies: array-like
            List of collision energies.
        l_vals: array-like
            List of partial waves.
        method: str
            One of 'numerov' or 'logderiv'.
        
        Returns
        -------
        S: ndarray
            S-matrix.
        sigma: ndarray
            Total integrated cross section.
        sigma_ch: ndarray
            Integrated partial wave cross sections.
        """
        # Prepare coupling matrix
        Vmat = np.zeros((len(self.r), len(l_vals), len(l_vals)))
        for l in range(len(l_vals)):
            V_l = self.V + (l*(l+1))/(2*self.mass*self.r**2)
            Vmat[:,l,l] = V_l
        
        # Solve coupled channels equations and calculate cross sections
        sigma_ch = np.zeros((len(l_vals), energies.size))
        for iE, E in enumerate(energies):
            if method == 'numerov':
                Y_N = numerov_asymptotic(E, self.mass, Vmat, self.r)
                shift = 0
            elif method == 'logderiv':
                Y_N = log_deriv(E, self.mass, Vmat, self.r)
                shift = 1
            else:
                raise ValueError('Invalid method')
            S = S_from_log_deriv(Y_N, E, l_vals, self.mass, V, self.r, shift=shift)
            for ch in range(len(l_vals)):
                k_ch = np.sqrt(2*mass*np.abs(E - V[-1, ch, ch]))
                sigma_ch[ch, iE] = np.pi * np.abs(1 - S[ch, ch])**2 / k_ch**2 * (2*l_vals[ch]+1)
        sigma = sigma_ch.sum(axis=0)
        
        return S, sigma, sigma_ch


def S_from_log_deriv(Y_N, E, l, m, V, r, shift=0):
    """Calculate S matrix from log-derivative matrix.
    """
    nch = len(l)
    N = r.size - 1
    n = N - 1 + shift# because of zero indexing
    Id = np.eye(nch) # Identity matrix

    # Following Krems:
    k = np.sqrt(np.diag(2.*m*(E - V[n])))
    j = np.zeros((nch, nch))
    jp = np.zeros((nch, nch))
    y = np.zeros((nch, nch))
    yp = np.zeros((nch, nch))
    for i in range(nch):
        riccati_jn_i, riccati_jn_ip = riccati_jn(max(l), k[i]*r[n])
        j[i, i] = 1/np.sqrt(k[i]) * riccati_jn_i[l[i]]
        jp[i, i] = np.sqrt(k[i]) * riccati_jn_ip[l[i]]
        riccati_yn_i, riccati_yn_ip = riccati_yn(max(l), k[i]*r[n])
        y[i, i] = 1/np.sqrt(k[i]) * riccati_yn_i[l[i]]
        yp[i, i] = np.sqrt(k[i]) * riccati_yn_ip[l[i]]
    K = np.linalg.inv(Y_N @ y - yp) @ (Y_N @ j - jp)
    S = (Id + 1j*K) @ np.linalg.inv(Id - 1j*K)

    return S


def numerov_asymptotic(E, m, V, r):
    """Renormalized Numerov method.
    
    Formulated in the asymptotic basis, based on lecture notes by Millard Alexander.
    """
    
    # Numerov propagation

    ## Prepare matrices
    nch = V.shape[-1]
    Id = np.eye(nch) # Identity matrix
    N = r.size - 1
    h = r[1] - r[0]
    # TODO: allow for nonequidistant grid

    U = np.zeros((N+1, nch, nch))
    T = np.zeros((N+1, nch, nch))
    R = np.zeros((N+1, nch, nch))

    ## First and second steop
    T[0] = -h**2/12. * 2*m*(E*Id - V[0])
    U[0] = 12.*np.linalg.inv(Id - T[0]) - 10.*Id
    R[1] = U[0]

    T[1] = -h**2/12. * 2*m*(E*Id - V[1])
    U[1] = 12.*np.linalg.inv(Id - T[1]) - 10.*Id
    R[2] = U[1] - np.linalg.inv(U[0])

    ## loop over remaining steps
    for i in range(2, N+1):
        T[i] = -h**2/12. * 2*m*(E*Id - V[i])
        U[i] = 12.*np.linalg.inv(Id - T[i]) - 10.*Id
        R[i] = U[i] - np.linalg.inv(R[i-1])

    # Calculate log derivative matrix Y_N
    n = N-1 # because of zero indexing
    Y_N = 1./h * (
         (0.5*Id-T[n+1]) @ np.linalg.inv(Id-T[n+1]) @ R[n+1]
        -(0.5*Id-T[n-1]) @ np.linalg.inv(Id-T[n-1]) @ np.linalg.inv(R[n])) @ (Id-T[n])

    return Y_N


def log_deriv_1ch(E, m, V, r):
    """Log derivative method for single-channel scattering.
    
    Implemented following the notation from Johnson JCP 67, 9, 1977.
    See also Krems' book, Sec. 8.1.6
    """
    N = r.size - 1
    assert N % 2 == 0, 'r must have an odd number of grid points'

    y = np.zeros(r.size)
    y[0] = 1e+30
    for i in range(1, r.size):
        h = r[i] - r[i-1] # Krems defined it the other way around... r[i+1] - r[i]
        Q = 2*m*(E - V[i])
        if i % 2 == 0:
            u_i = Q
            if i == N:
                w_i = 1
            else:
                w_i = 2
        else:
            u_i = Q / (1 + h*h/6 * Q)
            w_i = 4
        y[i] = y[i-1]/(1 + h*y[i-1]) - h/3*w_i*u_i

    return y[N]


def log_deriv(E, m, V, r):
    """Log derivative method.

    Implemented following the notation from Johnson JCP 67, 9, 1977.
    See also Krems' book, Sec. 8.1.6
    """
    N = r.size - 1
    assert N % 2 == 0, 'r must have an odd number of grid points'

    nch = V.shape[-1]
    Id = np.eye(nch) # Identity matrix
    Y = np.zeros((r.size, nch, nch))
    Y[0] = 1e+30 * Id
    for i in range(1, r.size):
        h = r[i] - r[i-1] # Krems defined it the other way around... r[i+1] - r[i]
        Q = 2*m*(E*Id - V[i])
        if i % 2 == 0:
            U_i = Q
            if i == N:
                w_i = 1
            else:
                w_i = 2
        else:
            U_i = np.linalg.inv(Id + h*h/6 * Q) @ Q
            w_i = 4
        Y[i] = np.linalg.inv(Id + h*Y[i-1]) @ Y[i-1] - h/3*w_i*U_i

    return Y[N]