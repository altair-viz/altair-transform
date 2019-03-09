import altair as alt
from altair_transform.vegaexpr import eval_vegajs


def confidence_interval(x, level):
    from scipy import stats
    return stats.t.interval(level, len(x)-1, loc=x.mean(), scale=x.sem())


AGG_REPLACEMENTS = {
    'argmin': 'idxmin',
    'argmax': 'idxmax',
    'average': 'mean',
    'ci0': lambda x: confidence_interval(x, 0.05),
    'ci1': lambda x: confidence_interval(x, 0.95),
    'distinct': 'nunique',
    'stderr': 'sem',
    'stdev': 'std',
    'stdevp': lambda x: x.std(ddof=0),
    'missing': lambda x: x.isnull().sum(),
    'q1': lambda x: x.quantile(0.25),
    'q3': lambda x: x.quantile(0.75),
    'valid': 'count',
    'values': 'count',
    'variance': 'var',
    'variancep': lambda x: x.var(ddof=0)
}


def apply_transform(df, transforms, inplace=False):
    if not inplace:
        df = df.copy()

    if isinstance(transforms, dict):
        transforms = [transforms]

    for transform in transforms:
        transform = alt.Transform.from_dict(transform)
        if isinstance(transform, alt.CalculateTransform):
            df[transform['as']] = df.apply(
                lambda datum: eval_vegajs(transform.calculate, datum),
                axis=1)
        elif isinstance(transform, alt.FilterTransform):
            if not isinstance(transform.filter, str):
                raise NotImplementedError("non-string filter")
            mask = df.apply(
                lambda datum: eval_vegajs(transform.filter, datum),
                axis=1).astype(bool)
            df = df[mask]
        elif isinstance(transform, alt.AggregateTransform):
            groupby = transform['groupby']
            for aggregate in transform['aggregate']:
                op = aggregate['op'].to_dict()
                field = aggregate['field']
                col = aggregate['as']

                op = AGG_REPLACEMENTS.get(op, op)

                if groupby is alt.Undefined:
                    df[col] = df[field].aggregate(op)
                else:
                    result = df.groupby(groupby)[field].aggregate(op)
                    result.name = col
                    df = df.join(result, on=groupby)
        else:
            raise NotImplementedError(
                f"{transform.__class__.__name__} not implemented.")

    return df
