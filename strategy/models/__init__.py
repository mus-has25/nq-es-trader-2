from strategy.models.ou_reversion import OUReversionModel
from strategy.models.vwap_reversion import VWAPReversionModel
from strategy.models.trend_cont import TrendContinuationModel
from strategy.models.liq_sweep_fvg import LiquiditySweepFVGModel

ALL_MODELS = [OUReversionModel, VWAPReversionModel, TrendContinuationModel,
              LiquiditySweepFVGModel]
