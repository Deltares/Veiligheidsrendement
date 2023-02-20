import pytest

from FloodDefenceSystem.Mechanisms import LSF_heave, LSF_sellmeijer


def test_LSF_heave():
    # set tests data
    r_exit, h, h_exit, d_cover, kwelscherm = (0.9, 6, 4, 2, 0)

    # calculate
    (g_h, i, i_c) = LSF_heave(r_exit, h, h_exit, d_cover, kwelscherm)

    # assert
    assert g_h == pytest.approx(-0.6)
    assert i == pytest.approx(0.9)
    assert i_c == pytest.approx(0.3)


def test_LSF_sellmeijer():
    #set tests data
    h, h_exit, d_cover, L, D, d70, k, mPiping = (7,4,2,50,30,.00075,1e-5,1.)

    #calculate
    (g_p, delta_h, delta_h_c) = LSF_sellmeijer(h, h_exit, d_cover, L, D, d70, k, mPiping)

    # assert
    assert g_p == pytest.approx(12.12568371313684)
    assert delta_h == pytest.approx(2.4)
    assert delta_h_c == pytest.approx(14.525683713136841)