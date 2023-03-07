from typing import Dict
from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
import logging

from vrtool.flood_defence_system.dike_traject import DikeTraject

class SmartTargetReliabilityStrategy(StrategyBase):
    def evaluate(self, traject: DikeTraject, solutions_dict: Dict[str, Solutions], splitparams=False):
        # TODO add a smarter OI version where the failure probability budget is partially redistributed over the mechanisms.

        # find section where it is most attractive to make 1 or multiple mechanisms to meet the cross sectional
        # reliability index
        # choice 1: geotechnical mechanisms ok for 2075

        # choice 2:also height ok for 2075
        logging.error("SmartTargetReliability is not implemented yet")
