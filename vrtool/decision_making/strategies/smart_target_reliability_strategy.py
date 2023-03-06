from vrtool.decision_making.strategies.strategy_base import StrategyBase
import logging

class SmartTargetReliabilityStrategy(StrategyBase):
    def evaluate(self, traject, solutions, splitparams=False):
        # TODO add a smarter OI version where the failure probability budget is partially redistributed over the mechanisms.

        # find section where it is most attractive to make 1 or multiple mechanisms to meet the cross sectional
        # reliability index
        # choice 1: geotechnical mechanisms ok for 2075

        # choice 2:also height ok for 2075
        logging.error("SmartTargetReliability is not implemented yet")
