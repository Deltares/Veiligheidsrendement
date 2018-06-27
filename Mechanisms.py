## This script contains limit state functions for the different mechanisms.
## It was translated from the scripts in Matlab Open Earth Tools that were used in the safety assessment

import numpy as np

def LSF_heave(Input):
    #lambd,h,h_b,d,i_ch
    if Input.d_cover_pip <= 0:      #geen deklaag = heave treedt altijd op
        i_c = 0
    elif Input.scherm == 'Ja':
        i_c = 0.5
    elif Input.scherm == 'Nee':
        i_c = 0.3
    else:
        print('The LSF of heave has no clue what to do')
    delta_phi = (Input.h - Input.h_exit) * Input.r_exit

    i = (delta_phi/Input.d_cover_pip) if Input.d_cover_pip > 0 else 99
    g_h = i_c - i
    return g_h, i, i_c

def LSF_sellmeijer(Input):
    delta_h_c = Input.m_Piping * sellmeijer2017(Input.L,Input.D,Input.theta,Input.d70,Input.k);    # Critical head difference (resistance):
    delta_h = Input.h - Input.h_exit - 0.3 * Input.d_cover_pip                                # Head difference (load)
    g_p = delta_h_c - delta_h                                                                 # Resistance minus load (incl. model factors)
    if delta_h<0:
        delta_h = 0
    return g_p, delta_h, delta_h_c

def LSF_uplift(Input):
    #lambd,h,h_b,d,gamma_sat,m_u
    m_u = 1.
    if Input.d_cover <= 0:                          #no cover layer so no uplift
        dh_c = 0
    else:
        dh_c= Input.d_cover * (Input.gamma_sat - Input.gamma_w) / Input.gamma_w;
    # print(dh_c)
    dh = (Input.h - Input.h_exit) * Input.r_exit
    g_u = m_u * dh_c - dh;  # Limit State Function
    return g_u, dh, dh_c
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

def zBligh(inp):
    h, phi_in, C, L = inp
    z_Bligh = (L/C) - (h-phi_in)
    return z_Bligh