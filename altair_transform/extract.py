"""Tools for extracting transforms from encodings"""
from collections import defaultdict
import copy
from typing import List, Tuple

import altair as alt


def extract_transform(chart: alt.Chart) -> alt.Chart:
    """Extract transforms within a chart specification."""
    chart = chart.copy()
    encoding_dict = chart.encoding.copy().to_dict(context={"data": chart.data})
    encoding, transform = _encoding_to_transform(encoding_dict)
    if transform:
        chart.encoding = alt.FacetedEncoding.from_dict(encoding)
        if chart.transform is alt.Undefined:
            chart.transform = []
        chart.transform.extend(alt.Transform.from_dict(t) for t in transform)
    return chart


def _encoding_to_transform(encoding: dict) -> Tuple[dict, List[dict]]:
    """Extract transforms from an encoding dict."""
    # TODO: what if one encoding has multiple keys? Is this valid?
    by_category: defaultdict = defaultdict(dict)
    new_encoding: dict = {}
    for enc, definition in encoding.items():
        for key in ["bin", "aggregate", "timeUnit"]:
            if key in definition:
                by_category[key][enc] = copy.deepcopy(definition)
                break
        else:
            new_encoding[enc] = copy.deepcopy(definition)

    groupby: List[str] = list(new_encoding.keys())
    transform: List[dict] = []

    if by_category["bin"]:
        raise NotImplementedError("Extracting bin transforms")

    if by_category["timeUnit"]:
        raise NotImplementedError("Extracting timeUnit transforms")

    agg_transforms: List[dict] = []
    for enc, definition in by_category["aggregate"].items():
        aggregate = definition.pop("aggregate")
        field = definition.pop("field", None)
        new_field = aggregate if field is None else f"{aggregate}_{field}"
        agg_dict = {"op": aggregate, "as": new_field}
        if field is not None:
            agg_dict["field"] = field
        agg_transforms.append(agg_dict)
        definition["field"] = new_field
        new_encoding[enc] = definition
    if agg_transforms:
        transform.append({"aggregate": agg_transforms, "groupby": groupby})

    # Sanity check
    assert encoding.keys() == new_encoding.keys()

    return new_encoding, transform
