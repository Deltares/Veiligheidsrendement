import numpy as np


## This script contains limit state functions for the different mechanisms.
## It was translated from the scripts in Matlab Open Earth Tools that were used in the safety assessment
def calculate_lsf_heave(r_exit, h, h_exit, d_cover, kwelscherm):
    # lambd,h,h_b,d,i_ch
    if isinstance(kwelscherm, str):
        if kwelscherm == "Ja":
            kwelscherm = 1
        elif kwelscherm == "Nee":
            kwelscherm = 0

    # For semiprob
    if d_cover <= 0:  # geen deklaag = heave treedt altijd op
        i_c = 0
    elif int(kwelscherm) == 1:
        i_c = 0.5
    elif int(kwelscherm) == 0:
        i_c = 0.3
    else:
        print("The LSF of heave has no clue what to do")

    # According to Formula Sander Kapinga,veilighiedsfactor heave
    if r_exit > 0:
        if h_exit > h:
            # hoog achterland dus geen piping
            i = 1e-50
        else:
            delta_phi = (h - h_exit) * r_exit
            i = (delta_phi / d_cover) if d_cover > 0 else 99
    else:
        # delta_phi = 0
        i_c = 0.5
        i = 0.1

    g_h = i_c - i

    return g_h, i, i_c


def calculate_lsf_sellmeijer(h, h_exit, d_cover, L, D, d70, k, mPiping):
    delta_h_c = mPiping * calculate_sellmeijer_2017(
        L, D, d70, k
    )  # Critical head difference (resistance):
    delta_h = h - h_exit - 0.3 * d_cover  # Head difference (load)
    g_p = delta_h_c - delta_h  # Resistance minus load (incl. model factors)
    if delta_h < 0:
        delta_h = 0

    return g_p, delta_h, delta_h_c


def calculate_lsf_uplift(r_exit, h, h_exit, d_cover, gamma_sat):
    gamma_w = 9.81
    # lambd,h,h_b,d,gamma_sat,m_u
    m_u = 1.0
    if d_cover <= 0:  # no cover layer so no uplift
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


def calculate_sellmeijer_2017(
    seepage_length: float,
    upper_sand_layer_thickness: float,
    particle_diameter: float,
    permeability: float,
) -> float:
    """
    Calculates the Sellmeijer 2017 formula.

    Args:
        seepage_length (float):  L - Seepage Length (48)
        upper_sand_layer_thickness (float): D - Thickness of upper sand layer (49)
        particle_diameter (float): d70 - Particle diameter (D70) (56)
        permeability (float):  k - Permeability of the upper sand layer (55)

    Returns:
        float: Delta HC
    """
    d70m = 2.08e-4
    # reference d70
    nu = 1.33e-6
    # dynamic viscosity of water at 10degC
    eta = 0.25
    # White's constant
    kappa = (nu / 9.81) * permeability
    theta = 37
    Fres = eta * (16.5 / 9.81) * np.tan(theta / 180 * np.pi)
    Fscale = (d70m / (kappa * seepage_length) ** (1 / 3)) * (
        (particle_diameter / d70m) ** 0.4
    )

    # F1        = 1.65 * eta * np.tan(theta/180*np.pi)**0.35;
    # F2        = d70m / (nu / 9.81 * k * L) ** (1/3) * (d70/d70m) ** 0.39;
    # F3        = 0.91 * (D/L)**(0.28/(((D/L)**2.8)-1)+0.04);
    if upper_sand_layer_thickness == seepage_length:
        Fgeometry = 1
    else:
        Fgeometry = 0.91 * (upper_sand_layer_thickness / seepage_length) ** (
            0.28 / (((upper_sand_layer_thickness / seepage_length) ** 2.8) - 1) + 0.04
        )

    delta_h_c = Fres * Fscale * Fgeometry * seepage_length
    # delta_h_c = F1 * F2 * F3 * L;
    return delta_h_c


def calculate_z_uplift(inp, mode: str = "Prob"):
    # if it is a dictionary: split according to names
    if isinstance(inp, dict):
        D = inp["D"]
        d_cover = inp["d_cover"]
        h_exit = inp["h_exit"]
        r_exit = inp["r_exit"]
        L = inp["Lvoor"] + inp["Lachter"]
        d70 = inp["d70"]
        k = inp["k"]
        gamma_sat = inp["gamma_sat"]
        kwelscherm = inp["kwelscherm"]
        mPiping = 1.0
        h_exit = h_exit - inp["dh_exit(t)"]
        h = inp["h"] + inp["dh"]
    # with ageing & water level change:
    else:
        if len(inp) == 13:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                dh_exit,
                h,
                dh,
            ) = inp
            h_exit = h_exit - dh_exit
            h = h + dh  # with ageing:
        if len(inp) == 12:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                dh_exit,
                h,
            ) = inp
            h_exit = h_exit - dh_exit
        # without ageing:
        elif len(inp) == 11:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                h,
            ) = inp
    g_u, dh_u, dhc_u = calculate_lsf_uplift(r_exit, h, h_exit, d_cover, gamma_sat)
    if mode == "Prob":
        return [g_u]
    else:
        return g_u, dh_u, dhc_u


def calculate_z_heave(inp, mode: str = "Prob"):
    # if it is a dictionary: split according to names
    if isinstance(inp, dict):
        D = inp["D"]
        d_cover = inp["d_cover"]
        h_exit = inp["h_exit"]
        r_exit = inp["r_exit"]
        L = inp["Lvoor"] + inp["Lachter"]
        d70 = inp["d70"]
        k = inp["k"]
        gamma_sat = inp["gamma_sat"]
        kwelscherm = inp["kwelscherm"]
        mPiping = 1.0
        h_exit = h_exit - inp["dh_exit(t)"]
        h = inp["h"] + inp["dh"]
    # with ageing & water level change:
    else:
        if len(inp) == 13:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                dh_exit,
                h,
                dh,
            ) = inp
            h_exit = h_exit - dh_exit
            h = h + dh  # with ageing:
        if len(inp) == 12:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                dh_exit,
                h,
            ) = inp
            h_exit = h_exit - dh_exit
        # without ageing:
        elif len(inp) == 11:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                h,
            ) = inp

    g_h, i, i_c = calculate_lsf_heave(r_exit, h, h_exit, d_cover, kwelscherm)
    if mode == "Prob":
        return [g_h]
    else:
        return g_h, i, i_c


def calculate_z_piping(inp, mode: str = "Prob"):
    # if it is a dictionary: split according to names
    if isinstance(inp, dict):
        D = inp["D"]
        d_cover = inp["d_cover"]
        h_exit = inp["h_exit"]
        r_exit = inp["r_exit"]
        L = inp["Lvoor"] + inp["Lachter"]
        d70 = inp["d70"]
        k = inp["k"]
        gamma_sat = inp["gamma_sat"]
        kwelscherm = inp["kwelscherm"]
        mPiping = 1.0  # inp['mPiping']
        h_exit = h_exit - inp["dh_exit(t)"]
        h = inp["h"] + inp["dh"]
    # with ageing & water level change:
    else:
        if len(inp) == 13:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                dh_exit,
                h,
                dh,
            ) = inp
            h_exit = h_exit - dh_exit
            h = h + dh  # with ageing:
        if len(inp) == 12:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                dh_exit,
                h,
            ) = inp
            h_exit = h_exit - dh_exit
        # without ageing:
        elif len(inp) == 11:
            (
                D,
                d_cover,
                h_exit,
                r_exit,
                L,
                d70,
                k,
                gamma_sat,
                kwelscherm,
                mPiping,
                h,
            ) = inp

    g_p, dh_p, dhc_p = calculate_lsf_sellmeijer(
        h, h_exit, d_cover, L, D, d70, k, mPiping
    )
    if mode == "Prob":
        return [g_p]
    else:
        return g_p, dh_p, dhc_p


def calculate_z_piping_total(inp):
    # with ageing & water level change:
    if len(inp) == 13:
        (
            D,
            d_cover,
            h_exit,
            r_exit,
            L,
            d70,
            k,
            gamma_sat,
            kwelscherm,
            mPiping,
            dh_exit,
            h,
            dh,
        ) = inp
        h_exit = h_exit - dh_exit
        h = h + dh
    # with ageing:
    if len(inp) == 12:
        (
            D,
            d_cover,
            h_exit,
            r_exit,
            L,
            d70,
            k,
            gamma_sat,
            kwelscherm,
            mPiping,
            dh_exit,
            h,
        ) = inp
        h_exit = h_exit - dh_exit
    # without ageing:
    elif len(inp) == 11:
        D, d_cover, h_exit, r_exit, L, d70, k, gamma_sat, kwelscherm, mPiping, h = inp
    # r_exit, h, h_exit,d,i_ch, gamma_sat,m_u,L,D,theta,d70,k,m_p = inp

    g_h, i, i_c = calculate_lsf_heave(r_exit, h, h_exit, d_cover, kwelscherm)
    g_p, dh_p, dhc_p = calculate_lsf_sellmeijer(
        h, h_exit, d_cover, L, D, d70, k, mPiping
    )
    g_u, dh_u, dhc_u = calculate_lsf_uplift(r_exit, h, h_exit, d_cover, gamma_sat)
    z_piping = max(g_p, g_u, g_h)
    # import pdb; pdb.set_trace()
    return [z_piping]
