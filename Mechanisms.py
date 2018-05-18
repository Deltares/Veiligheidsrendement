## This script contains limit state functions for the different mechanisms.
## It was translated from the scripts in Matlab Open Earth Tools that were used in the safety assessment

import numpy as np

def LSF_heave(lambd,h,h_b,d,i_ch):
    if d <= 0:
        [g_h] = deal(999);
    else:
        m_phi = 1; #KW: this is assumed equal to 1!!! considered as a multiplicative variable (and not additive)
        phi = h_b + m_phi * (h - h_b) * lambd; #Head at Exit Point
        i   = (phi - h_b) / d;
        g_h = i_ch - i;                        #Limit State Function
    return g_h

def LSF_sellmeijer(h,h_b,d,L,D,theta,d70,k,m_p):
    delta_h_c = m_p * sellmeijer2017(L,D,theta,d70,k);    # Critical head difference (resistance):
    delta_h = h - h_b - 0.3 * d                           # Head difference (load)
    g_p = delta_h_c - delta_h                             # Resistance minus load (incl. model factors)
    if delta_h<0:
        delta_h = 0
    return g_p, delta_h, delta_h_c
def sellmeijer2017(L,D,theta,d70,k):

    #L     - Seepage Length (48)
    #D     - Thickness of upper sand layer (49)
    #theta - Bedding angle (Theta) (52)
    #d70   - Particle diameter (D70) (56)
    #k     - Permeability of the upper sand layer (55)
    RD        = 0.725;   # RD (set equal to the reference RD to neglect RD-influence)
    RDm       = 0.725;   # reference RD
    d70m      = 2.08e-4; # reference d70
    nu        = 1.33e-6; # dynamic viscosity of water at 10degC
    eta       = 0.25;    # White's constant
    kappa     = (nu / 9.81) * k
    # Fres      = eta*(16.5/9.81)*np.tan(theta/180*np.pi)
    Fres      = eta*(14.8/9.81)*np.tan(theta/180*np.pi)

    Fscale    = (d70m/(kappa*L) ** (1/3))*((d70/d70m) ** 0.4)

    # F1        = 1.65 * eta * np.tan(theta/180*np.pi) * (RD/RDm)**0.35;
    # F2        = d70m / (nu / 9.81 * k * L) ** (1/3) * (d70/d70m) ** 0.39;
    # F3        = 0.91 * (D/L)**(0.28/(((D/L)**2.8)-1)+0.04);
    if D==L:
        Fgeometry = 1
    else:
        Fgeometry = 0.91 * (D / L) ** (0.28 / (((D / L) ** 2.8) - 1) + 0.04);

    delta_h_c = Fres * Fscale * Fgeometry * L
    # delta_h_c = F1 * F2 * F3 * L;
    return delta_h_c

def LSF_uplift(lambd,h,h_b,d,gamma_sat,m_u):
    if d <= 0:
        [g_u] = deal(999);
    phi = h_b + (h - h_b) * lambd;                    #Head at Exit Point
    delta_phi_c = d * (gamma_sat - 10) / 10;           #Critical Head Difference
    g_u = m_u * delta_phi_c - (phi - h_b);             #Limit State Function
    h_cu = h_b + (m_u * delta_phi_c) / lambd;        #Critical Water Level (for Z = 0)
    return g_u
def zPiping(inp):
    lambd, h, h_b,d,i_ch, gamma_sat,m_u,L,D,theta,d70,k,m_p = inp
    g_h = LSF_heave(lambd,h,h_b,d,i_ch)
    g_p = LSF_sellmeijer(h,h_b,d,L,D,theta,d70,k,m_p)
    g_u = LSF_uplift(lambd,h,h_b,d,gamma_sat,m_u)

    z_piping = max(g_h,g_p,g_u)
    #import pdb; pdb.set_trace()
    return [z_piping]
def zOverflow(inp):
    h, h_c = inp
    z_overflow = h_c-h
    return [z_overflow]
def zTotal(inp):
    lambd, h, h_b,d,i_ch, gamma_sat,m_u,L,D,theta,d70,k,m_p,h_c = inp
    inp_piping = lambd, h, h_b,d,i_ch, gamma_sat,m_u,L,D,theta,d70,k,m_p
    inp_overfl = h, h_c
    z_piping = zPiping(inp_piping)
    z_overflow = zOverflow(inp_overfl)
    return min(z_piping,z_overflow)