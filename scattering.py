import numpy as np
from scipy.special import riccati_jn, riccati_yn, spherical_jn, spherical_yn, hankel1, hankel2


# Taken from the notebook `test_numerov_for_inelastic_scattering.ipynb` from
# `/Projects/elastic_scattering/numerov_test/alexander_twochannel`
# Based on the Skript by Millard Alexander
def numerov_asymptotic(E, m, V, r):        
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
def numerov_asymptotic_k(k, m, V, r):        
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
    T[0] = -h**2/12. * k**2
    U[0] = 12.*np.linalg.inv(Id - T[0]) - 10.*Id
    R[1] = U[0]

    T[1] = -h**2/12. * k**2
    U[1] = 12.*np.linalg.inv(Id - T[1]) - 10.*Id
    R[2] = U[1] - np.linalg.inv(U[0])

    ## loop over remaining steps
    for i in range(2, N+1):
        T[i] = -h**2/12. * k**2
        U[i] = 12.*np.linalg.inv(Id - T[i]) - 10.*Id
        R[i] = U[i] - np.linalg.inv(R[i-1])

    # Calculate log derivative matrix Y_N
    n = N-1 # because of zero indexing
    Y_N = 1./h * (
         (0.5*Id-T[n+1]) @ np.linalg.inv(Id-T[n+1]) @ R[n+1]
        -(0.5*Id-T[n-1]) @ np.linalg.inv(Id-T[n-1]) @ np.linalg.inv(R[n])) @ (Id-T[n])

    return Y_N


def S_from_log_deriv(Y_N, E, l, m, V, r, shift=0):

    nch = len(l)
    N = r.size - 1
    n = N - 1 + shift# because of zero indexing
    Id = np.eye(nch) # Identity matrix

    # Calculate scattering matrix S

    ## NOTE: this is only valid for l=0!
    #k = np.sqrt(np.diag(2.*m*(E - V[n])))
    #h1 = np.diag(-1.j/np.sqrt(k) * np.exp(+1.j*k*r[n]))
    #h2 = h1.conj()
    #h1p = +1.j*np.diag(k) * h1
    #h2p = -1.j*np.diag(k) * h2
    #S = np.linalg.inv(h1p - Y_N @ h1) @ (h2p - Y_N @ h2)

    ## for l != 0, Following the Krems:
    ## NOTE: there may be an issue with the sign convention on how to define y1p and y2p.
    ## See the dicsussion above.
    #k = np.sqrt(np.diag(2.*m*(E - V[n])))
    #j = np.zeros((nch, nch))
    #jp = np.zeros((nch, nch))
    #y = np.zeros((nch, nch))
    #yp = np.zeros((nch, nch))
    #for i in range(nch):
    #    riccati_jn_i, riccati_jn_ip = riccati_jn(max(l), k[i]*r[n])
    #    j[i, i] = 1/np.sqrt(k[i]) * riccati_jn_i[l[i]]
    #    jp[i, i] = np.sqrt(k[i]) * riccati_jn_ip[l[i]]
    #    riccati_yn_i, riccati_yn_ip = riccati_yn(max(l), k[i]*r[n])
    #    y[i, i] = -1/np.sqrt(k[i]) * riccati_yn_i[l[i]]
    #    yp[i, i] = -np.sqrt(k[i]) * riccati_yn_ip[l[i]]
    #K = np.linalg.inv(Y_N @ j - jp) @ (Y_N @ y - yp)
    #S = (Id + 1j*K) @ np.linalg.inv(Id - 1j*K)

    # Strictly following Krems:
    # This seems to be the correct one!
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
def S_from_log_deriv_k(Y_N, k, l, m, V, r, shift=0):

    nch = len(l)
    N = r.size - 1
    n = N - 1 + shift# because of zero indexing
    Id = np.eye(nch) # Identity matrix

    # Calculate scattering matrix S

    ## NOTE: this is only valid for l=0!
    #k = np.sqrt(np.diag(2.*m*(E - V[n])))
    #h1 = np.diag(-1.j/np.sqrt(k) * np.exp(+1.j*k*r[n]))
    #h2 = h1.conj()
    #h1p = +1.j*np.diag(k) * h1
    #h2p = -1.j*np.diag(k) * h2
    #S = np.linalg.inv(h1p - Y_N @ h1) @ (h2p - Y_N @ h2)

    ## for l != 0, Following the Krems:
    ## NOTE: there may be an issue with the sign convention on how to define y1p and y2p.
    ## See the dicsussion above.
    #k = np.sqrt(np.diag(2.*m*(E - V[n])))
    #j = np.zeros((nch, nch))
    #jp = np.zeros((nch, nch))
    #y = np.zeros((nch, nch))
    #yp = np.zeros((nch, nch))
    #for i in range(nch):
    #    riccati_jn_i, riccati_jn_ip = riccati_jn(max(l), k[i]*r[n])
    #    j[i, i] = 1/np.sqrt(k[i]) * riccati_jn_i[l[i]]
    #    jp[i, i] = np.sqrt(k[i]) * riccati_jn_ip[l[i]]
    #    riccati_yn_i, riccati_yn_ip = riccati_yn(max(l), k[i]*r[n])
    #    y[i, i] = -1/np.sqrt(k[i]) * riccati_yn_i[l[i]]
    #    yp[i, i] = -np.sqrt(k[i]) * riccati_yn_ip[l[i]]
    #K = np.linalg.inv(Y_N @ j - jp) @ (Y_N @ y - yp)
    #S = (Id + 1j*K) @ np.linalg.inv(Id - 1j*K)

    # Strictly following Krems:
    # This seems to be the correct one!
    j = np.zeros((nch, nch))
    jp = np.zeros((nch, nch))
    y = np.zeros((nch, nch))
    yp = np.zeros((nch, nch))
    for i in range(nch):
        riccati_jn_i, riccati_jn_ip = riccati_jn(max(l), k*r[n])
        j[i, i] = 1/np.sqrt(k) * riccati_jn_i[l[i]]
        jp[i, i] = np.sqrt(k) * riccati_jn_ip[l[i]]
        riccati_yn_i, riccati_yn_ip = riccati_yn(max(l), k*r[n])
        y[i, i] = 1/np.sqrt(k) * riccati_yn_i[l[i]]
        yp[i, i] = np.sqrt(k) * riccati_yn_ip[l[i]]
    K = np.linalg.inv(Y_N @ y - yp) @ (Y_N @ j - jp)
    S = (Id + 1j*K) @ np.linalg.inv(Id - 1j*K)

    return S


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
def log_deriv_k(k, m, V, r):
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
        Q = k**2 * Id
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