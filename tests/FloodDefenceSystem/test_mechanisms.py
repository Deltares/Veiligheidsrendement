import pytest

from vrtool.FloodDefenceSystem.Mechanisms import LSF_heave, LSF_sellmeijer


class TestMechanisms:
    def test_lsf_heave(self):
        # set tests data
        r_exit = 0.9
        h = 6
        h_exit = 4
        d_cover = 2
        kwelscherm = 0

        # calculate
        (g_h, i, i_c) = LSF_heave(r_exit, h, h_exit, d_cover, kwelscherm)

        # assert
        assert g_h == pytest.approx(-0.6)
        assert i == pytest.approx(0.9)
        assert i_c == pytest.approx(0.3)

    def test_lsf_sellmeijer(self):
        # set tests data
        h = 7
        h_exit = 4
        d_cover = 2
        l = 50
        d = 30
        d70 = 0.00075
        k = 1e-5
        m_piping = 1.0

        # calculate
        (g_p, delta_h, delta_h_c) = LSF_sellmeijer(
            h, h_exit, d_cover, l, d, d70, k, m_piping
        )

        # assert
        assert g_p == pytest.approx(12.12568371313684)
        assert delta_h == pytest.approx(2.4)
        assert delta_h_c == pytest.approx(14.525683713136841)
