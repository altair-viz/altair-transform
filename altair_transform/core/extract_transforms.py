"""Tools for extracting transforms from encodings"""
from collections import defaultdict
import copy
from typing import List, Tuple


def _encoding_to_transform(encoding: dict) -> Tuple[dict, List[dict]]:
    """Given an encoding dict, extract transforms and update the encoding."""
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
        field = definition.pop("field")
        new_field = f"{aggregate}_{field}"
        agg_transforms.append({"op": aggregate, "field": field, "as": new_field})
        definition["field"] = new_field
        new_encoding[enc] = definition
    transform.append({"aggregate": agg_transforms, "groupby": groupby})

    # Sanity check
    assert encoding.keys() == new_encoding.keys()

    return new_encoding, transform
