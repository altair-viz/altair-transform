import pytest

import numpy as np
import pandas as pd

from altair.utils.data import to_values
from altair_transform import apply
from altair_transform.core.aggregate import AGG_REPLACEMENTS


AGGREGATES = ['argmax', 'argmin', 'average', 'count', 'distinct',
              'max', 'mean',  'median', 'min', 'missing', 'q1',
              'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp',
              'sum', 'valid', 'values', 'variance', 'variancep']

AGG_SKIP = ['ci0', 'ci1']  # These require scipy.


FILTER_PREDICATES = [
  ('datum.x < datum.y',
   lambda df: df[df.x < df.y]),
  ({'not': 'datum.i < 5'},
   lambda df: df[~(df.i < 5)]),
  ({'and': [{'field': 'x', 'lt': 50}, {'field': 'i', 'gte': 2}]},
   lambda df: df[(df.x < 50) & (df.i >= 2)]),
  ({'or': [{'field': 'y', 'gt': 50}, {'field': 'i', 'lte': 4}]},
   lambda df: df[(df.y > 50) | (df.i <= 4)]),
  ({'field': 'c', 'oneOf': ['A', 'B']},
   lambda df: df[df.c.isin(['A', 'B'])]),
  ({'field': 'x', 'range': [30, 60]},
   lambda df: df[(df.x >= 30) & (df.x <= 60)]),
  ({'field': 'c', 'equal': 'B'},
   lambda df: df[df.c == 'B']),
]


@pytest.fixture
def data():
    rand = np.random.RandomState(42)
    return pd.DataFrame({
        'x': rand.randint(0, 100, 12),
        'y': rand.randint(0, 100, 12),
        'i': range(12),
        'c': list('AAABBBCCCDDD'),
    })


def test_calculate_transform(data):
    transform = {'calculate': 'datum.x + datum.y', 'as': 'z'}
    out1 = apply(data, transform)

    out2 = data.copy()
    out2['z'] = data.x + data.y

    assert out1.equals(out2)


@pytest.mark.parametrize("filter,calc", FILTER_PREDICATES)
def test_filter_transform(data, filter, calc):
    out1 = apply(data, {'filter': filter})
    out2 = calc(data)
    assert out1.equals(out2)


@pytest.mark.parametrize('groupby', [True, False])
@pytest.mark.parametrize('op', set(AGGREGATES) - set(AGG_SKIP))
def test_aggregate_transform(data, groupby, op):
    field = 'x'
    col = 'z'
    group = 'c'

    transform = {'aggregate': [{'op': op, 'field': field, 'as': col}]}
    if groupby:
        transform['groupby'] = [group]

    op = AGG_REPLACEMENTS.get(op, op)
    out = apply(data, transform)

    def validate(group):
        return np.allclose(group[field].aggregate(op), group[col])

    if groupby:
        assert out.groupby(group).apply(validate).all()
    else:
        assert validate(out)


@pytest.mark.parametrize('lookup_key', ['c', 'c2'])
def test_lookup_transform(data, lookup_key):
    lookup = pd.DataFrame({
        lookup_key: list('ABCD'),
        'z': [3, 1, 4, 5]
    })
    transform = {
        'lookup': 'c',
        'from': {
            'data': to_values(lookup),
            'key': lookup_key,
            'fields': ['z']
        }
    }
    out1 = apply(data, transform)
    out2 = pd.merge(data, lookup, left_on='c', right_on=lookup_key)
    if lookup_key != 'c':
        out2 = out2.drop(lookup_key, axis=1)
    assert out1.equals(out2)


@pytest.mark.parametrize('lookup_key', ['c', 'c2'])
@pytest.mark.parametrize('default', [None, 'missing'])
def test_lookup_transform_default(data, lookup_key, default):
    lookup = pd.DataFrame({
        lookup_key: list('ABC'),
        'z': [3, 1, 4]
    })
    transform = {
        'lookup': 'c',
        'from': {
            'data': to_values(lookup),
            'key': lookup_key,
            'fields': ['z']
        }
    }
    if default is not None:
        transform['default'] = default

    out = apply(data, transform)
    undef = (out['c'] == 'D')
    if default is None:
        assert out.loc[undef, 'z'].isnull().all()
    else:
        assert (out.loc[undef, 'z'] == default).all()


def test_bin_transform(data):
    transform = {'bin': True, 'field': 'x', 'as': 'xbin'}
    out = apply(data, transform)
    assert 'xbin' in out.columns

    transform = {'bin': True, 'field': 'x', 'as': ['xbin1', 'xbin2']}
    out = apply(data, transform)
    assert 'xbin1' in out.columns
    assert 'xbin2' in out.columns


def test_window_transform_basic(data):
    transform = {
        'window': [{'op': 'sum', 'field': 'x', 'as': 'xsum'}],
    }
    out = apply(data, transform)
    expected = data['x'].cumsum()
    assert out['xsum'].equals(expected.astype(float))


def test_window_transform_sorted(data):
    transform = {
        'window': [{'op': 'sum', 'field': 'x', 'as': 'xsum'}],
        'sort': [{'field': 'x'}]
    }
    out = apply(data, transform)
    expected = data['x'].sort_values().cumsum().sort_index()
    assert out['xsum'].equals(expected.astype(float))


def test_window_transform_grouped(data):
    transform = {
        'window': [{'op': 'sum', 'field': 'x', 'as': 'xsum'}],
        'groupby': ['y'],
    }
    out = apply(data, transform)
    expected = data.groupby('y').rolling(len(data), min_periods=1)
    expected = expected['x'].sum().reset_index('y', drop=True).sort_index()
    assert out['xsum'].equals(expected)


def test_multiple_transforms(data):
    transform = [
        {'calculate': '0.5 * (datum.x + datum.y)', 'as': 'xy_mean'},
        {'filter': 'datum.x < datum.xy_mean'}
    ]
    out1 = apply(data, transform)
    out2 = data.copy()
    out2['xy_mean'] = 0.5 * (data.x + data.y)
    out2 = out2[out2.x < out2.xy_mean]

    assert out1.equals(out2)
