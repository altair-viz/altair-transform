# altair-transform

Python evaluation of Altair/Vega-Lite transforms.

[![build status](http://img.shields.io/travis/altair-viz/altair-transform/master.svg)](https://travis-ci.org/altair-viz/altair-transform)
[![code style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

``altair-transform`` requires Python 3.6 or later. Install with:

    $ pip install altair_transform

Altair-transform evaluates [Altair](http://altair-viz.github.io) and [Vega-Lite](http://vega.github.io/vega-lite)
transforms directly in Python. This can be useful in a number of contexts, illustrated in the examples below.

## Example: Extracting Data

The Vega-Lite specification includes the ability to apply a
wide range of transformations to input data within the chart
specification. As an example, here is a sliding window average
of a Gaussian random walk, implemented in Altair:

```python
import altair as alt
import numpy as np
import pandas as pd

rand = np.random.RandomState(12345)

df = pd.DataFrame({
    'x': np.arange(200),
    'y': rand.randn(200).cumsum()
})

points = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q'
)

line = alt.Chart(df).transform_window(
    ymean='mean(y)',
    sort=[alt.SortField('x')],
    frame=[5, 5]
).mark_line(color='red').encode(
    x='x:Q',
    y='ymean:Q'
)

points + line
```
![Altair Visualization](https://raw.githubusercontent.com/altair-viz/altair-transform/master/images/random_walk.png)

Because the transform is encoded within the renderer, however, the
computed values are not directly accessible from the Python layer.

This is where ``altair_transform`` comes in. It includes a (nearly)
complete Python implementation of Vega-Lite's transform layer, so
that you can easily extract a pandas dataframe with the computed
values shown in the chart:

```python
from altair_transform import extract_data
data = extract_data(line)
data.head()
```
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>ymean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.204708</td>
      <td>0.457749</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.274236</td>
      <td>0.771093</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.245203</td>
      <td>1.041320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.800933</td>
      <td>1.336943</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.164847</td>
      <td>1.698085</td>
    </tr>
  </tbody>
</table>

From here, you can work with the transformed data directly
in Python.

## Example: Pre-Aggregating Large Datasets

Altair creates chart specifications containing the full dataset.
The advantage of this is that the data used to make the chart is entirely transparent; the disadvantage is that it causes issues as datasets grow large.
To prevent users from inadvertently crashing their browsers by trying to send too much data to the frontend, Altair limits the data size by default.
For example, a histogram of 20000 points:

```python
import altair as alt
import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame({
    'x': np.random.randn(20000)
})
chart = alt.Chart(df).mark_bar().encode(
    alt.X('x', bin=True),
    y='count()'
)
chart
```
```pyerr
MaxRowsError: The number of rows in your dataset is greater than the maximum allowed (5000). For information on how to plot larger datasets in Altair, see the documentation
```
There are several possible ways around this, as mentioned in Altair's [FAQ](https://altair-viz.github.io/user_guide/faq.html#maxrowserror-how-can-i-plot-large-datasets).
Altiar-transform provides another option via the ``transform_chart()`` function, which will pre-transform the data according to the chart specification, so that the final chart specification holds the aggregated data rather than the full dataset:
```python
from altair_transform import transform_chart
new_chart = transform_chart(chart)
new_chart
```
![Altair Visualization](https://raw.githubusercontent.com/altair-viz/altair-transform/master/images/histogram.png)

Examining the new chart specification, we can see that it contains the pre-aggregated dataset:
```python
new_chart.data
```
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_binned</th>
      <th>x_binned2</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-4.0</td>
      <td>-3.0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.0</td>
      <td>-2.0</td>
      <td>444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.0</td>
      <td>-1.0</td>
      <td>2703</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>6815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>6858</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2706</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>423</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>22</td>
    </tr>
  </tbody>
</table>

## Limitations

``altair_transform`` currently works only for non-compound charts; that is, it cannot transform or extract data from layered, faceted, repeated, or concatenated charts.

There are also a number of less-used transform options that are not yet fully supported. These should explicitly raise a ``NotImplementedError`` if you attempt to use them.
