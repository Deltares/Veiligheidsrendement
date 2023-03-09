import shelve
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import seaborn as sns

from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.post_processing.plot_lcc import plot_lcc


class TestPlotLcc:
    @pytest.mark.skip(
        reason="TODO: This code needs to be adapted to new architecture and data structures."
    )
    def test_plot_lcc_with_valid_data(self):
        """
        This test is a representation of the previous `def main()` contained in the file `plot_lcc.py`
        """
        # 1. Define test data.
        _config = VrtoolConfig()

        # initialize the case that we consider. We start with a small one, eventually we will use a big one.
        ##PLOT SETTINGS
        t_0 = 2025
        rel_year = t_0 - 2025

        def get_from_shelve(output_file: Path) -> dict:
            _shelve = shelve.open(str(output_file))
            _shelve_dict = {}
            for _key in _shelve:
                _shelve_dict[_key] = _shelve[_key]
            _shelve.close()
            return _shelve_dict

        _loaded_traject: DikeTraject = get_from_shelve(
            _config.output_directory / "AfterStep1.out"
        )
        _loaded_solutions = get_from_shelve(_config.output_directory / "AfterStep2.out")
        _loaded_strategies: list[StrategyBase] = get_from_shelve(
            _config.output_directory / "FINALRESULT.out"
        )
        _greedy_mode = "Optimal"
        # greedy_mode = 'SatisfiedStandard'

        _loaded_strategies[0].getSafetyStandardStep(_loaded_traject.GeneralInfo["Pmax"])
        _loaded_strategies[1].makeSolution(
            _config.output_directory.joinpath(
                "results", "FinalMeasures_Doorsnede-eisen.csv"
            ),
            type="Final",
        )
        _loaded_strategies[0].makeSolution(
            _config.output_directory.joinpath(
                "results", "FinalMeasures_Veiligheidsrendement.csv"
            ),
            step=_loaded_strategies[0].SafetyStandardStep,
            type="SatisfiedStandard",
        )

        # pane 1: reliability in the relevant years
        # left: system reliability
        # right: all sections
        figsize = (12, 2)
        # color settings
        optimized_colors = {
            "n_colors": 6,
            "start": 1.5,
            "rot": 0.3,
            "gamma": 1.5,
            "hue": 1.0,
            "light": 0.8,
            "dark": 0.3,
        }
        targetrel_colors = {
            "n_colors": 6,
            "start": 0.5,
            "rot": 0.3,
            "gamma": 1.5,
            "hue": 1.0,
            "light": 0.8,
            "dark": 0.3,
        }
        case_settings = {
            "directory": _config.output_directory,
            "language": _config.language,
            "beta_or_prob": _config.beta_or_prob,
        }
        for plot_t in _config.assessment_plot_years:
            plot_year = str(plot_t + t_0)
            _loaded_traject.plotAssessment(
                fig_size=figsize,
                t_list=[plot_t],
                labels_limited=True,
                system_rel=True,
                show_xticks=True,
                case_settings=case_settings,
                custom_name="Assessment_" + plot_year + ".png",
                title_in="(a) \n"
                + r"$\bf{Predicted~reliability~in~"
                + plot_year
                + "}$",
            )
            #
            # #pane 2: reliability in 2075, with Greedy optimization
            _loaded_traject.plotAssessment(
                fig_size=figsize,
                t_list=[plot_t],
                labels_limited=True,
                system_rel=True,
                case_settings=case_settings,
                custom_name="GreedyStrategy_" + plot_year + ".png",
                reinforcement_strategy=_loaded_strategies[0],
                greedymode=_greedy_mode,
                show_xticks=True,
                title_in="(c)\n"
                + r"$\bf{Optimized~investment}$ - Reliability in "
                + plot_year,
                colors=optimized_colors,
            )
            #
            # #pane 3: reliability in 2075, with Target Reliability Approach
            _loaded_traject.plotAssessment(
                fig_size=figsize,
                t_list=[plot_t],
                labels_limited=True,
                system_rel=True,
                case_settings=case_settings,
                custom_name="TargetReliability_" + plot_year + ".png",
                reinforcement_strategy=_loaded_strategies[1],
                show_xticks=True,
                title_in="(e) \n"
                + r"$\bf{Target~reliability~based~investment}$ -  Reliability in "
                + plot_year,
                colors=targetrel_colors,
            )
        #
        # pane 4: measures per dike section for Greedy
        _loaded_strategies[0].plotMeasures(
            traject=_loaded_traject,
            PATH=_config.output_directory,
            fig_size=figsize,
            crestscale=25.0,
            show_xticks=True,
            flip=True,
            greedymode=_greedy_mode,
            title_in="(b) \n" + r"$\bf{Greedy strategy}$ - Measures",
            colors=optimized_colors,
        )
        # #pane 5: measures per dike section for Target
        #
        _loaded_strategies[1].plotMeasures(
            traject=_loaded_traject,
            PATH=_config.output_directory,
            fig_size=figsize,
            crestscale=25.0,
            show_xticks=True,
            flip=True,
            title_in="(d) \n"
            + r"$\bf{Target~reliability~based~investment}$ - Measures",
            colors=targetrel_colors,
        )

        # #pane 6: Investment costs per dike section for both
        twoColors = [
            sns.cubehelix_palette(**optimized_colors)[1],
            sns.cubehelix_palette(**targetrel_colors)[1],
        ]
        plot_lcc(
            _loaded_strategies,
            _loaded_traject,
            output_dir=_config.output_directory,
            fig_size=figsize,
            flip=True,
            greedymode=_greedy_mode,
            title_in="(f) \n" + r"$\bf{LCC~of~both~approaches}$",
            color=twoColors,
        )

        # LCC-beta for t=50
        for t_plot in [0, 50]:
            for cost_type in ["Initial", "LCC"]:
                MeasureTable = StrategyBase.get_measure_table(
                    _loaded_solutions, language="EN", abbrev=True
                )
                figsize = (6, 4)
                plt.figure(102, figsize=figsize)
                _loaded_strategies[0].plotBetaCosts(
                    _loaded_traject,
                    save_dir=_config.output_directory,
                    t=t_plot,
                    cost_type=cost_type,
                    fig_id=102,
                    markersize=10,
                    final_step=_loaded_strategies[0].OptimalStep,
                    color=twoColors[0],
                    series_name="Optimized investment",
                    MeasureTable=MeasureTable,
                    final_measure_symbols=False,
                )
                _loaded_strategies[1].plotBetaCosts(
                    _loaded_traject,
                    save_dir=_config.output_directory,
                    t=t_plot,
                    cost_type=cost_type,
                    fig_id=102,
                    markersize=10,
                    color=twoColors[1],
                    series_name="Target reliability based investment",
                    MeasureTable=MeasureTable,
                    last=True,
                    final_measure_symbols=True,
                )
                plt.savefig(
                    _config.output_directory.joinpath(
                        "Priority order Beta vs LCC_" + str(t_plot + t_0) + ".png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                    format="png",
                )
                plt.close()
