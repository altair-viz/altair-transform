import pandas as pd
import altair as alt

from typing import Union, Optional

DataType = Union[dict, pd.Series, alt.SchemaBase]
ChartType = Union[dict, alt.SchemaBase]


def to_dataframe(data: DataType,
                 context: Optional[ChartType] = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data

    if not isinstance(data, dict):
        data = data.to_dict()

    if 'values' in data:
        return pd.DataFrame(data['values'])

    if 'url' in data:
        url = data['url']
        fmt = data.get('format', url.split('.')[-1])
        if fmt == 'csv':
            return pd.read_csv(url)
        elif fmt == 'json':
            return pd.read_json(url)
        else:
            raise ValueError(f"Unknown format for UrlData: '{fmt}'")

    # TODO: implement named data with context
    data = alt.Data.from_dict(data)
    raise NotImplementedError(f"Data of type {type(data)}")
