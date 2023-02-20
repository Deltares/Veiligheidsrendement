import numpy as np
from scipy import interpolate

import config
import ProbabilisticTools.ProbabilisticFunctions as ProbabilisticFunctions


## This script contains limit state functions for the different mechanisms.
## It was translated from the scripts in Matlab Open Earth Tools that were used in the safety assessment
def OverflowHRING(input, year, mode="assessment", Pt=None):
    """year is relative to start year. input contains relevant inputs"""
    if mode == "assessment":
        h_t = input["h_crest"] - input["d_crest"] * (year)
        years = input["hc_beta"].columns.values.astype(np.int32)
        betas = []
        for j in years:
            betas.append(interpolate.interp1d(input['hc_beta'].index.values,input['hc_beta'][str(j)],fill_value='extrapolate')(h_t))
        beta = interpolate.interp1d(years,betas,fill_value='extrapolate')(year+config.t_0)
        return beta, ProbabilisticFunctions.beta_to_pf(beta)
    if mode == 'design':
        t_beta_interp = interpolate.interp2d(input['hc_beta'].columns.values.astype(np.float32), input['hc_beta'].index.values,input['hc_beta'],bounds_error=False)
        h_grid = np.linspace(input['hc_beta'].index.values.min(),input['hc_beta'].index.values.max(),50)
        h_beta = t_beta_interp(year+config.t_0,h_grid).flatten()
        new_crest = interpolate.interp1d(h_beta,h_grid,fill_value='extrapolate')(ProbabilisticFunctions.pf_to_beta(Pt)).item()
        return new_crest, ProbabilisticFunctions.pf_to_beta(Pt)
def OverflowSimple(h_crest, q_crest, h_c, q_c, beta, mode='assessment', Pt=None, design_variable=None,iterative_solve = False,beta_t = False):
    if mode == 'assessment':
        if q_c[0] != q_c[-1:]:
            beta_hc = interpolate.interp2d(h_c, q_c, beta, kind='linear', fill_value='extrapolate')
            beta = np.min([beta_hc(h_crest, q_crest), 8.])
        else:
            beta_hc = interpolate.interp1d(h_c, beta, kind='linear', fill_value='extrapolate')
            beta = np.min([beta_hc(h_crest), [8.]])
        Pf = ProbabilisticFunctions.beta_to_pf(beta)
        if not iterative_solve:
            return beta, Pf
        else:
            return beta - beta_t
    elif mode == 'design':
        beta_t = ProbabilisticFunctions.pf_to_beta(Pt)
        if design_variable == 'h_crest':
            if q_c[0] != q_c[-1:]:
                beta_hc = interpolate.interp2d(beta, q_c, h_c, kind='linear', fill_value='extrapolate')
                h_crest = beta_hc(beta_t, q_crest)
            else:
                beta_hc = interpolate.interp1d(beta, h_c, kind='linear', fill_value='extrapolate')
                h_crest = beta_hc(beta_t)
            return h_crest, beta_t
        pass

def LSF_heave(r_exit, h, h_exit, d_cover, kwelscherm):
    #lambd,h,h_b,d,i_ch
    if isinstance(kwelscherm, str):
        if kwelscherm == 'Ja':
            kwelscherm = 1
        elif kwelscherm == 'Nee':
            kwelscherm = 0

    #For semiprob
    if d_cover <= 0:      #geen deklaag = heave treedt altijd op
        i_c = 0
    elif int(kwelscherm) == 1:
        i_c = 0.5
    elif int(kwelscherm) == 0:
        i_c = 0.3
    else:
        print('The LSF of heave has no clue what to do')

    #According to Formula Sander Kapinga,veilighiedsfactor heave
    if r_exit > 0:
        if h_exit > h:
            #hoog achterland dus geen piping
            i=1e-50
        else:
            delta_phi = (h - h_exit) * r_exit
            i = (delta_phi/d_cover) if d_cover > 0 else 99
    else:
        # delta_phi = 0
        i_c = 0.5
        i = 0.1

    g_h = i_c - i

    return g_h, i, i_c

def LSF_sellmeijer(h, h_exit, d_cover, L, D, d70, k, mPiping):
    delta_h_c = mPiping * sellmeijer2017(L, D, d70, k)  # Critical head difference (resistance):
    delta_h = h - h_exit - 0.3 * d_cover                                # Head difference (load)
    g_p = delta_h_c - delta_h                                                                 # Resistance minus load (incl. model factors)
    if delta_h < 0:
        delta_h = 0

    return g_p, delta_h, delta_h_c

def LSF_uplift(r_exit, h, h_exit, d_cover, gamma_sat):
    gamma_w = 9.81
    #lambd,h,h_b,d,gamma_sat,m_u
    m_u = 1.
    if d_cover <= 0:                          #no cover layer so no uplift
        dh_c = 0
    else:
        dh_c = d_cover * (gamma_sat - gamma_w) / gamma_w
    # print(dh_c)
    dh = (h - h_exit) * r_exit

    # According to Formula Sander Kapinga,veilighiedsfactor openbarsten
    if dh <= 0:
        dh_c = 0.5
        dh = 0.1

    g_u = m_u * dh_c - dh  # Limit State Function

    return g_u, dh, dh_c

def sellmeijer2017(L,D,d70,k):

    #L     - Seepage Length (48)
    #D     - Thickness of upper sand layer (49)
    #theta - Bedding angle (Theta) (52)
    #d70   - Particle diameter (D70) (56)
    #k     - Permeability of the upper sand layer (55)

    d70m      = 2.08e-4; # reference d70
    nu        = 1.33e-6; # dynamic viscosity of water at 10degC
    eta       = 0.25;    # White's constant
    kappa     = (nu / 9.81) * k
    theta     = 37
    Fres      = eta*(16.5/9.81)*np.tan(theta/180*np.pi)
    Fscale    = (d70m/(kappa*L) ** (1/3))*((d70/d70m) ** 0.4)

    # F1        = 1.65 * eta * np.tan(theta/180*np.pi)**0.35;
    # F2        = d70m / (nu / 9.81 * k * L) ** (1/3) * (d70/d70m) ** 0.39;
    # F3        = 0.91 * (D/L)**(0.28/(((D/L)**2.8)-1)+0.04);
    if D == L:
        Fgeometry = 1
    else:
        Fgeometry = 0.91 * (D / L) ** (0.28 / (((D / L) ** 2.8) - 1) + 0.04)

    delta_h_c = Fres * Fscale * Fgeometry * L
    # delta_h_c = F1 * F2 * F3 * L;
    return delta_h_c

def zUplift(inp, mode='Prob'):
    #if it is a dictionary: split according to names
    if isinstance(inp, dict):
        D = inp['D']; d_cover = inp['d_cover']; h_exit = inp['h_exit']; r_exit = inp['r_exit']
        L = inp['Lvoor']+inp['Lachter']; d70 = inp['d70']; k = inp['k']; gamma_sat = inp['gamma_sat']
        kwelscherm = inp['kwelscherm']; mPiping = 1.
        h_exit = h_exit - inp['dh_exit(t)']
        h = inp['h'] + inp['dh']
    #with ageing & water level change:
    else:
        if len(inp) == 13:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h, dh = inp
            h_exit = h_exit - dh_exit
            h = h + dh    #with ageing:
        if len(inp) == 12:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h = inp
            h_exit = h_exit - dh_exit
        #without ageing:
        elif len(inp) == 11:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, h = inp
    g_u, dh_u, dhc_u = LSF_uplift(r_exit, h, h_exit, d_cover, gamma_sat)
    if mode == 'Prob':
        return [g_u]
    else:
        return g_u, dh_u, dhc_u

def zHeave(inp, mode='Prob'):
    #if it is a dictionary: split according to names
    if isinstance(inp, dict):
        D = inp['D']; d_cover = inp['d_cover']; h_exit = inp['h_exit']; r_exit = inp['r_exit']
        L = inp['Lvoor']+inp['Lachter']; d70 = inp['d70']; k = inp['k']; gamma_sat = inp['gamma_sat']
        kwelscherm = inp['kwelscherm']; mPiping = 1.
        h_exit = h_exit - inp['dh_exit(t)']
        h = inp['h'] + inp['dh']
    #with ageing & water level change:
    else:
        if len(inp) == 13:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h, dh = inp
            h_exit = h_exit - dh_exit
            h = h + dh    #with ageing:
        if len(inp) == 12:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h = inp
            h_exit = h_exit - dh_exit
        #without ageing:
        elif len(inp) == 11:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, h = inp

    g_h, i, i_c = LSF_heave(r_exit, h, h_exit, d_cover, kwelscherm)
    if mode == 'Prob':
        return [g_h]
    else:
        return g_h, i, i_c

def zPiping(inp, mode='Prob'):
    #if it is a dictionary: split according to names
    if isinstance(inp, dict):
        D = inp['D']; d_cover = inp['d_cover']; h_exit = inp['h_exit']; r_exit = inp['r_exit']
        L = inp['Lvoor']+inp['Lachter']; d70 = inp['d70']; k = inp['k']; gamma_sat = inp['gamma_sat']
        kwelscherm = inp['kwelscherm']; mPiping = 1. #inp['mPiping']
        h_exit = h_exit - inp['dh_exit(t)']
        h = inp['h'] + inp['dh']
    #with ageing & water level change:
    else:
        if len(inp) == 13:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h, dh = inp
            h_exit = h_exit - dh_exit
            h = h + dh    #with ageing:
        if len(inp) == 12:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h = inp
            h_exit = h_exit - dh_exit
        #without ageing:
        elif len(inp) == 11:
            D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, h = inp

    g_p, dh_p, dhc_p = LSF_sellmeijer(h, h_exit, d_cover, L, D, d70, k, mPiping)
    if mode == 'Prob':
        return [g_p]
    else:
        return g_p, dh_p, dhc_p

def zPipingTotal(inp):
    #with ageing & water level change:
    if len(inp) == 13:
        D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h, dh = inp
        h_exit = h_exit - dh_exit
        h = h + dh
    #with ageing:
    if len(inp) == 12:
        D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, dh_exit, h = inp
        h_exit = h_exit - dh_exit
    #without ageing:
    elif len(inp) == 11:
        D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, h = inp
    # r_exit, h, h_exit,d,i_ch, gamma_sat,m_u,L,D,theta,d70,k,m_p = inp

    g_h, i, i_c = LSF_heave(r_exit, h, h_exit, d_cover, kwelscherm)
    g_p, dh_p, dhc_p = LSF_sellmeijer(h, h_exit, d_cover, L, D, d70, k, mPiping)
    g_u, dh_u, dhc_u = LSF_uplift(r_exit, h, h_exit, d_cover, gamma_sat)
    z_piping = max(g_p, g_u, g_h)
    #import pdb; pdb.set_trace()
    return [z_piping]

def zOverflow(inp):
    if len(inp) == 4:
        h_c, dh_c, h, dh = inp
        h = h + dh
    #with ageing:
    elif len(inp) == 3:
        h_c, dh_c, h = inp
    elif len(inp) == 2:
        h_c, h = inp
        dh_c = 0
    z = (h_c - dh_c) - h
    return [z]

def simpleLSF(input):
    R,dR,S = input
    return [(R-dR)-S]