import altair as alt
import pandas as pd
from .visitor import visit


@visit.register
def visit_timeunit(transform: alt.TimeUnitTransform, df: pd.DataFrame):
    raise NotImplementedError("TimeUnitTransform")
