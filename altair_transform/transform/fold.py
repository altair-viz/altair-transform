import altair as alt
import pandas as pd
from .visitor import visit


@visit.register(alt.FoldTransform)
def visit_fold(transform: alt.FoldTransform, df: pd.DataFrame) -> pd.DataFrame:
    transform = transform.to_dict()
    fold = transform["fold"]
    var_name, value_name = transform.get("as", ("key", "value"))
    value_vars = [c for c in df.columns if c in fold]
    id_vars = [c for c in df.columns if c not in fold]

    # Add an index to track input order
    dfi = df.reset_index(drop=True).reset_index()
    index_name = dfi.columns[0]
    melted = dfi.melt(
        id_vars=[index_name] + id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
    return (
        pd.merge(melted, dfi, on=[index_name] + id_vars, how="left")
        .sort_values(index_name)
        .drop(index_name, axis=1)
        .reset_index(drop=True)
    )
