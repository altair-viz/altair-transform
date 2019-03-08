import altair as alt
from altair_transform.vegaexpr import eval_vegajs

def apply_transform(df, transforms, inplace=False):
    if not inplace:
        df = df.copy()

    for transform in transforms:
        transform = alt.Transform.from_dict(transform)
        if isinstance(transform, alt.CalculateTransform):
            func = lambda datum: eval_vegajs(transform.calculate, datum)
            df[transform['as']] = df.apply(func, axis=1)
        elif isinstance(transform, alt.FilterTransform):
            if not isinstance(transform.filter, str):
                raise NotImplementedError("non-string filter")
            func = lambda datum: eval_vegajs(transform.filter, datum)
            mask = df.apply(func, axis=1).astype(bool)
            df = df[mask]
        else:
            raise NotImplementedError(
                f"{transform.__class__.__name__} not implemented.")

    return df
