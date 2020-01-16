"""Python ports of vega utilities"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import math


# subdivide up to accuracy of 0.1 degrees
MIN_RADIANS = 0.1 * math.pi / 180

Number = Union[int, float]


def calculate_bins(
    extent: Tuple[Number, Number],
    anchor: Optional[Number] = None,
    base: Number = 10,
    divide: List[Number] = [5, 2],
    maxbins: Number = 10,
    minstep: Number = 0,
    nice: bool = True,
    step: Optional[Number] = None,
    steps: Optional[List[Number]] = None,
    span: Optional[Number] = None,
) -> np.ndarray:
    """Calculate the bins for a given dataset.

    This is a Python translation of the Javascript function available at
    https://github.com/vega/vega/blob/v5.9.1/packages/vega-statistics/src/bin.js

    Parameters
    ----------
    extent: Tuple[Number, Number]
        A two-element ([min, max]) array indicating the range of desired bin values.
    anchor: Number
        A value in the binned domain at which to anchor the bins, shifting the bin boundaries
        if necessary to ensure that a boundary aligns with the anchor value.
        Default value: the minimum bin extent value
    base: Number
        The number base to use for automatic bin determination (default is base 10).
        Default value: 10
    divide: List[Number]
        Scale factors indicating allowable subdivisions. The default value is [5, 2],
        which indicates that for base 10 numbers (the default base), the method may
        consider dividing bin sizes by 5 and/or 2. For example, for an initial step
        size of 10, the method can check if bin sizes of 2 (= 10/5), 5 (= 10/2),
        or 1 (= 10/(5*2)) might also satisfy the given constraints.
        Default value: [5, 2]
    maxbins: Number
        Maximum number of bins.
        Default value: 10
    minstep: Number
        A minimum allowable step size (particularly useful for integer values).
    nice: boolean
        If true, attempts to make the bin boundaries use human-friendly boundaries,
        such as multiples of ten.
        Default value: True
    step: Number
        An exact step size to use between bins.
        Note: If provided, options such as maxbins will be ignored.
    steps: List[Number]
        An array of allowable step sizes to choose from.

    Returns
    -------
    bins : numpy.ndarray
        array of bin edges.
    """
    start, stop, step = _bin(
        extent=extent,
        base=base,
        divide=divide,
        maxbins=maxbins,
        minstep=minstep,
        nice=nice,
        step=step,
        steps=steps,
        span=span,
    )

    N = math.ceil((stop - start) / step)

    if anchor is not None:
        start += anchor - (start + step * math.floor((anchor - start) / step))

    return start + step * np.arange(N + 1)


def _bin(
    extent: Tuple[Number, Number],
    base: Number = 20,
    divide: List[Number] = [5, 2],
    maxbins: Number = 10,
    minstep: Number = 0,
    nice: bool = True,
    step: Optional[Number] = None,
    steps: Optional[List[Number]] = None,
    span: Optional[Number] = None,
) -> Tuple[Number, Number, Number]:
    """Calculate the bins for a given dataset.

    This is a Python translation of the Javascript function available at
    https://github.com/vega/vega/blob/v5.9.1/packages/vega-statistics/src/bin.js
    """
    min_, max_ = extent
    assert max_ > min_
    span = span or (max_ - min_) or abs(min_) or 1
    logb = math.log(base)

    if step is not None:
        # If step is provided, we use it.
        pass
    elif steps is not None:
        # If steps provided, limit choice to acceptable sizes.
        v = span / maxbins
        steps = [step for step in steps if step < v]
        step = max(steps) if steps else steps[0]
    else:
        # Otherwise use span to determine step size.
        level = math.ceil(math.log(maxbins) / logb)
        step = max(minstep, pow(base, round(math.log(span) / logb) - level))

        # increase step size if too many bins
        while math.ceil(span / step) > maxbins:
            step *= base

        # decrease step size if allowed
        for div in divide:
            v = step / div
            if v >= minstep and span / v <= maxbins:
                step = v

    # update precision of min_ and max_
    v = math.log(step)
    precision = 0 if v >= 0 else math.floor(-v / logb) + 1
    eps = pow(base, -precision - 1)
    if nice:
        v = math.floor(min_ / step + eps) * step
        min_ = v - step if min_ < v else v
        max_ = math.ceil(max_ / step) * step

    start = min_
    stop = max_ if max_ != min_ else min_ + step
    return start, stop, step


def adaptive_sample(
    f: Callable[[np.ndarray], np.ndarray],
    extent: Tuple[float, float],
    min_steps: int = 25,
    max_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive sampling of a function.

    This is a Python translation of the Javascript function available at
    https://github.com/vega/vega/blob/v5.9.1/packages/vega-statistics/src/sampleCurve.js

    Parameters
    ----------
    f : callable
        Function to be adaptively sampled
    extent : tuple
        The extent of the sampling
    min_steps : int
        The minimum number of steps to consider
    max_steps : int
        The maximum number of steps to consider

    Returns
    -------
    x, y : np.ndarray
        The sampled function
    """

    min_x, max_x = extent
    span = max_x - min_x
    stop = span / max_steps

    # sample minimum points on uniform grid
    x = min_x + (np.arange(min_steps + 1) / min_steps) * span
    y = f(x)

    if min_steps == max_steps:
        # no adaptation, sample uniform grid directly and return
        return x, y

    # move on to perform adaptive refinement
    start_grid = list(zip(x, y))
    prev, next_ = start_grid[:1], start_grid[:0:-1]

    while next_:
        p0, p1 = prev[-1], next_[-1]

        # midpoint for potential curve subdivision
        xm = (p0[0] + p1[0]) / 2
        pm = (xm, f(xm))

        if pm[0] - p0[0] >= stop and _angleDelta(p0, pm, p1) > MIN_RADIANS:
            # maximum resolution has not yet been met, and
            # subdivision midpoint sufficiently different from endpoint
            # save subdivision, push midpoint onto the visitation stack
            next_.append(pm)
        else:
            # subdivision midpoint sufficiently similar to endpoint
            # skip subdivision, store endpoint, move to next point on the stack
            prev.append(p1)
            next_.pop()
    out = np.array(prev)
    return out[:, 0], out[:, 1]


def _angleDelta(
    p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]
) -> float:
    a0 = np.arctan2(r[1] - p[1], r[0] - p[0])
    a1 = np.arctan2(q[1] - p[1], q[0] - p[0])
    return abs(a0 - a1)
