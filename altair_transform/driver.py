"""Extract transformed data directly via a selenium webdriver."""
import io
import json
from typing import Any, Dict, List, Optional, Union

import altair as alt
import pandas as pd

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONDict = Dict[str, JSON]

CDN_URL = "https://cdn.jsdelivr.net/npm/{package}@{version}"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Embedding Vega-Lite</title>
  <script src="{vega_url}"></script>
  <script src="{vegalite_url}"></script>
  <script src="{vegaembed_url}"></script>
</head>
<body>
  <div id="vis"></div>
</body>
</html>
"""

EXTRACT_CODE = """
var spec = arguments[0];
var name = arguments[1];
var done = arguments[2];

vegaEmbed("#vis", spec, {"mode": "vega-lite"})
  .then(result => done({data: JSON.stringify(result.view.data(name))}))
  .catch(error => done({error: error.toString()}));
"""


def _serialize(df: pd.DataFrame) -> JSONDict:
    """Serialize a dataframe to a JSON dict."""
    return json.loads(df.to_json(orient="table"))


def _load(serialized: JSONDict) -> pd.DataFrame:
    """Load a dataframe from a JSON dict."""
    return pd.read_json(io.StringIO(json.dumps(serialized)), orient="table")


def _extract_data(spec: JSONDict, name: str = "data_0") -> pd.DataFrame:
    """Extract named data from a Vega-Lite chart spec.

    Parameters
    ----------
    spec : dict
        The Vega-Lite specification containing the data to extract

    name : string
        The name of the data stream to extract

    Returns
    -------
    data : pd.DataFrame
        The extracted data
    """
    # Optional deps
    from selenium.common.exceptions import NoSuchElementException
    from altair_saver import SeleniumSaver
    from altair_viewer import get_bundled_script

    js_resources = {
        "vega.js": get_bundled_script("vega", alt.VEGA_VERSION),
        "vega-lite.js": get_bundled_script("vega-lite", alt.VEGALITE_VERSION),
        "vega-embed.js": get_bundled_script("vega-embed", alt.VEGAEMBED_VERSION),
    }
    html = HTML_TEMPLATE.format(
        vega_url="/vega.js",
        vegalite_url="/vega-lite.js",
        vegaembed_url="/vega-embed.js",
    )

    url = SeleniumSaver._serve(html, js_resources)
    driver_name = SeleniumSaver._select_webdriver(20)
    driver = SeleniumSaver._registry.get(driver_name, 20)

    driver.get("about:blank")
    driver.get(url)

    try:
        driver.find_element_by_id("vis")
    except NoSuchElementException:
        raise RuntimeError(f"Could not load {url}")

    data = driver.execute_async_script(EXTRACT_CODE, spec, name)

    if "error" in data:
        raise ValueError(f"Javascript Error: {data['error']}")

    return pd.DataFrame.from_records(json.loads(data["data"]))


def apply(
    df: pd.DataFrame,
    transform: Union[
        None, JSONDict, alt.Transform, List[Union[JSONDict, alt.Transform]]
    ] = None,
) -> pd.DataFrame:
    """Extract transformed data from a Javascript rendering.

    Parameters
    ----------
    df : pd.DataFrame
    transform : list|dict
        A transform specification or list of transform specifications.
        Each specification must be valid according to Altair's transform
        schema.

    Returns
    -------
    df_transformed : pd.DataFrame
        The transformed dataframe.
    """
    if transform is None:
        transform = []
    elif not isinstance(transform, list):
        transform = [transform]
    chart = alt.Chart(df).mark_point()._add_transform(*transform)
    with alt.data_transformers.enable(max_rows=None, consolidate_datasets=False):
        spec = chart.to_dict()
    return _extract_data(spec, "data_0")


def get_tz_code() -> str:
    """Get the timezone code used by chromedriver."""
    # Optional deps
    from selenium.common.exceptions import NoSuchElementException
    from altair_saver import SeleniumSaver

    html = """<html><body><div id="vis"></div></body></html>"""
    script = "arguments[0](Intl.DateTimeFormat().resolvedOptions().timeZone)"
    url = SeleniumSaver._serve(html, {})
    driver_name = SeleniumSaver._select_webdriver(20)
    driver = SeleniumSaver._registry.get(driver_name, 20)
    driver.get("about:blank")
    driver.get(url)
    try:
        driver.find_element_by_id("vis")
    except NoSuchElementException:
        raise RuntimeError(f"Could not load {url}")
    return driver.execute_async_script(script)


def get_tz_offset(tz: Optional[str] = None) -> pd.Timedelta:
    """Get the timezone offset between Python and Javascript for dates with the given timezone.

    Parameters
    ----------
    tz : string (optional)
        The timezone of the input dates

    Returns
    -------
    offset : pd.Timedelta
        The offset between the Javasript representation and the Python representation
        of a date with the given timezone.
    """
    ts = pd.to_datetime("2012-01-01").tz_localize(tz)
    df = pd.DataFrame({"t": [ts]})
    out = apply(df, {"timeUnit": "year", "field": "t", "as": "year"})

    date_in = df.t[0]
    date_out = pd.to_datetime(1e6 * out.t)[0].tz_localize(tz)

    return date_out - date_in
