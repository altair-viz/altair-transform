"""
Evaluate vega expressions language
"""
import datetime as dtmodule
import math
import random
import sys
import pandas as pd
import time
from typing import Pattern

from altair_transform.utils import evaljs


def eval_vegajs(expression, datum=None):
    """Evaluate a vega expression"""
    namespace = {"datum": datum} if datum is not None else {}
    namespace.update(VEGAJS_NAMESPACE)
    return evaljs(expression, namespace)


# Type Coercion Functions
def isArray(value):
    """Returns true if value is an array, false otherwise."""
    return isinstance(value, list),


def isBoolean(value):
    """Returns true if value is a boolean (true or false), false otherwise."""
    return isinstance(value, bool)


def isDate(value):
    """Returns true if value is a Date object, false otherwise.

    This method will return false for timestamp numbers or
    date-formatted strings; it recognizes Date objects only.
    """
    return isinstance(value, dtmodule.datetime)


def isNumber(value):
    """Returns true if value is a number, false otherwise.

    NaN and Infinity are considered numbers.
    """
    return isinstance(value, (int, float))


def isObject(value):
    """Returns true if value is an object, false otherwise.

    Following JavaScript typeof convention, null values are considered objects.
    """
    return value is None or isinstance(value, dict)


def isRegExp(value):
    """
    Returns true if value is a RegExp (regular expression)
    object, false otherwise.
    """
    return isinstance(value, Pattern)


def isString(value):
    """Returns true if value is a string, false otherwise."""
    return isinstance(value, str)


# Type Coercion Functions
def toBoolean(value):
    """
    Coerces the input value to a boolean.
    Null values and empty strings are mapped to null.
    """
    return bool(value)


def toDate(value):
    """
    Coerces the input value to a Date instance.
    Null values and empty strings are mapped to null.
    If an optional parser function is provided, it is used to
    perform date parsing, otherwise Date.parse is used.
    """
    if value is None or value == "":
        return None
    return pd.to_datetime(value)


def toNumber(value):
    """
    Coerces the input value to a number.
    Null values and empty strings are mapped to null.
    """
    if value is None or value == "":
        return None
    return float(value)


def toString(value):
    """
    Coerces the input value to a string.
    Null values and empty strings are mapped to null.
    """
    if value is None:
        return None
    return str(value)


# Date/Time Functions
def now():
    """Returns the timestamp for the current time."""
    return round(time.time() * 1000, 0)


def datetime(year, month, day=0, hour=0, min=0, sec=0, millisec=0):
    """Returns a new Date instance.
    The month is 0-based, such that 1 represents February.
    """
    # TODO: do we need a local timezone?
    return dtmodule.datetime(year, month + 1, day, hour,
                             min, sec, millisec * 1000)


def date(datetime):
    """
    Returns the day of the month for the given datetime value, in local time.
    """
    return datetime.day


def day(datetime):
    """
    Returns the day of the week for the given datetime value, in local time.
    """
    return (datetime.weekday() + 1) % 7


def year(datetime):
    """Returns the year for the given datetime value, in local time."""
    return datetime.year


def quarter(datetime):
    """
    Returns the quarter of the year (0-3) for the given datetime value,
    in local time.
    """
    return (datetime.month - 1) // 3


def month(datetime):
    """
    Returns the (zero-based) month for the given datetime value, in local time.
    """
    return datetime.month - 1


def hours(datetime):
    """
    Returns the hours component for the given datetime value, in local time.
    """
    return datetime.hour


def minutes(datetime):
    """
    Returns the minutes component for the given datetime value, in local time.
    """
    return datetime.minute


def seconds(datetime):
    """
    Returns the seconds component for the given datetime value, in local time.
    """
    return datetime.second


def milliseconds(datetime):
    """
    Returns the milliseconds component for the given datetime value,
    in local time.
    """
    return datetime.microsecond / 1000


def time(datetime):
    """Returns the epoch-based timestamp for the given datetime value."""
    return datetime.timestamp() * 1000


def timezoneoffset(datetime):
    # TODO: use tzlocal?
    raise NotImplementedError()


def utc(year, month, day=0, hour=0, min=0, sec=0, millisec=0):
    """
    Returns a timestamp for the given UTC date.
    The month is 0-based, such that 1 represents February.
    """
    return dtmodule.datetime(year, month + 1, day, hour, min, sec,
                             millisec * 1000, tzinfo=pytz.utc)

# # utcdate(datetime)
# Returns the day of the month for the given datetime value, in UTC time.

# # utcday(datetime)
# Returns the day of the week for the given datetime value, in UTC time.

# # utcyear(datetime)
# Returns the year for the given datetime value, in UTC time.

# # utcquarter(datetime)
# Returns the quarter of the year (0-3) for the given datetime value, in UTC time.

# # utcmonth(datetime)
# Returns the (zero-based) month for the given datetime value, in UTC time.

# # utchours(datetime)
# Returns the hours component for the given datetime value, in UTC time.

# # utcminutes(datetime)
# Returns the minutes component for the given datetime value, in UTC time.

# # utcseconds(datetime)
# Returns the seconds component for the given datetime value, in UTC time.

# # utcmilliseconds(datetime)
# Returns the milliseconds component for the given datetime value, in UTC time.

# Back to Top

# Array Functions
# Functions for working with arrays of values.

# # extent(array)
# Returns a new [min, max] array with the minimum and maximum values of the input array, ignoring null, undefined, and NaN values.

# # clampRange(range, min, max)
# Clamps a two-element range array in a span-preserving manner. If the span of the input range is less than (max - min) and an endpoint exceeds either the min or max value, the range is translated such that the span is preserved and one endpoint touches the boundary of the [min, max] range. If the span exceeds (max - min), the range [min, max] is returned.

# # indexof(array, value)
# Returns the first index of value in the input array.

# # inrange(value, range)
# Tests whether value lies within (or is equal to either) the first and last values of the range array.

# # lastindexof(array, value)
# Returns the last index of value in the input array.

# # length(array)
# Returns the length of the input array.

# # lerp(array, fraction)
# Returns the linearly interpolated value between the first and last entries in the array for the provided interpolation fraction (typically between 0 and 1). For example, lerp([0, 50], 0.5) returns 25.

# # peek(array)
# Returns the last element in the input array. Similar to the built-in Array.pop method, except that it does not remove the last element. This method is a convenient shorthand for array[array.length - 1].

# # sequence([start, ]stop[, step])
# Returns an array containing an arithmetic sequence of numbers. If step is omitted, it defaults to 1. If start is omitted, it defaults to 0. The stop value is exclusive; it is not included in the result. If step is positive, the last element is the largest start + i * step less than stop; if step is negative, the last element is the smallest start + i * step greater than stop. If the returned array would contain an infinite number of values, an empty range is returned. The arguments are not required to be integers.

# # slice(array, start[, end])
# Returns a section of array between the start and end indices. If the end argument is negative, it is treated as an offset from the end of the array (length(array) + end).

# # span(array)
# Returns the span of array: the difference between the last and first elements, or array[array.length-1] - array[0].

# Back to Top

# String Functions
# Functions for modifying text strings.

# # indexof(string, substring)
# Returns the first index of substring in the input string.

# # lastindexof(string, substring)
# Returns the last index of substring in the input string.

# # length(string)
# Returns the length of the input string.

# # lower(string)
# Transforms string to lower-case letters.

# # pad(string, length[, character, align])
# Pads a string value with repeated instances of a character up to a specified length. If character is not specified, a space (‘ ‘) is used. By default, padding is added to the end of a string. An optional align parameter specifies if padding should be added to the 'left' (beginning), 'center', or 'right' (end) of the input string.

# # parseFloat(string)
# Parses the input string to a floating-point value. Same as JavaScript’s parseFloat.

# # parseInt(string)
# Parses the input string to an integer value. Same as JavaScript’s parseInt.

# # replace(string, pattern, replacement)
# Returns a new string with some or all matches of pattern replaced by a replacement string. The pattern can be a string or a regular expression. If pattern is a string, only the first instance will be replaced. Same as JavaScript’s String.replace.

# # slice(string, start[, end])
# Returns a section of string between the start and end indices. If the end argument is negative, it is treated as an offset from the end of the string (length(string) + end).

# # split(string, separator[, limit]) ≥ 4.3
# Returns an array of tokens created by splitting the input string according to a provided separator pattern. The result can optionally be constrained to return at most limit tokens.

# # substring(string, start[, end])
# Returns a section of string between the start and end indices.

# # truncate(string, length[, align, ellipsis])
# Truncates an input string to a target length. The optional align argument indicates what part of the string should be truncated: 'left' (the beginning), 'center', or 'right' (the end). By default, the 'right' end of the string is truncated. The optional ellipsis argument indicates the string to use to indicate truncated content; by default the ellipsis character … (\u2026) is used.

# # upper(string)
# Transforms string to upper-case letters.

# Back to Top

# Object Functions
# Functions for manipulating object instances.

# # merge(object1[, object2, …]) ≥ 4.0
# Merges the input objects object1, object2, etc into a new output object. Inputs are visited in sequential order, such that key values from later arguments can overwrite those from earlier arguments. Example: merge({a:1, b:2}, {a:3}) -> {a:3, b:2}.

# Back to Top

# Formatting Functions
# Functions for formatting number and datetime values as strings.

# # dayFormat(day)
# Formats a (0-6) weekday number as a full week day name, according to the current locale. For example: dayFormat(0) -> "Sunday".

# # dayAbbrevFormat(day)
# Formats a (0-6) weekday number as an abbreviated week day name, according to the current locale. For example: dayAbbrevFormat(0) -> "Sun".

# # format(value, specifier)
# Formats a numeric value as a string. The specifier must be a valid d3-format specifier (e.g., format(value, ',.2f').

# # monthFormat(month)
# Formats a (zero-based) month number as a full month name, according to the current locale. For example: monthFormat(0) -> "January".

# # monthAbbrevFormat(month)
# Formats a (zero-based) month number as an abbreviated month name, according to the current locale. For example: monthAbbrevFormat(0) -> "Jan".

# # timeFormat(value, specifier)
# Formats a datetime value (either a Date object or timestamp) as a string, according to the local time. The specifier must be a valid d3-time-format specifier. For example: timeFormat(timestamp, '%A').

# # timeParse(string, specifier)
# Parses a string value to a Date object, according to the local time. The specifier must be a valid d3-time-format specifier. For example: timeParse('June 30, 2015', '%B %d, %Y').

# # utcFormat(value, specifier)
# Formats a datetime value (either a Date object or timestamp) as a string, according to UTC time. The specifier must be a valid d3-time-format specifier. For example: utcFormat(timestamp, '%A').

# # utcParse(value, specifier)
# Parses a string value to a Date object, according to UTC time. The specifier must be a valid d3-time-format specifier. For example: utcParse('June 30, 2015', '%B %d, %Y').

# Back to Top

# RegExp Functions
# Functions for creating and applying regular expressions.

# # regexp(pattern[, flags]) - Creates a regular expression instance from an input pattern string and optional flags. Same as JavaScript’s RegExp.

# # test(regexp[, string]) - Evaluates a regular expression regexp against the input string, returning true if the string matches the pattern, false otherwise. For example: test(/\\d{3}/, "32-21-9483") -> true.

# Back to Top

# Color Functions
# Functions for representing colors in various color spaces. Color functions return objects that, when coerced to a string, map to valid CSS RGB colors.

# # rgb(r, g, b[, opacity]) | rgb(specifier)
# Constructs a new RGB color. If r, g and b are specified, these represent the channel values of the returned color; an opacity may also be specified. If a CSS Color Module Level 3 specifier string is specified, it is parsed and then converted to the RGB color space. Uses d3-color’s rgb function.

# # hsl(h, s, l[, opacity]) | hsl(specifier)
# Constructs a new HSL color. If h, s and l are specified, these represent the channel values of the returned color; an opacity may also be specified. If a CSS Color Module Level 3 specifier string is specified, it is parsed and then converted to the HSL color space. Uses d3-color’s hsl function.

# # lab(l, a, b[, opacity]) | lab(specifier)
# Constructs a new CIE LAB color. If l, a and b are specified, these represent the channel values of the returned color; an opacity may also be specified. If a CSS Color Module Level 3 specifier string is specified, it is parsed and then converted to the LAB color space. Uses d3-color’s lab function.

# # hcl(h, c, l[, opacity]) | hcl(specifier)
# Constructs a new HCL (hue, chroma, luminance) color. If h, c and l are specified, these represent the channel values of the returned color; an opacity may also be specified. If a CSS Color Module Level 3 specifier string is specified, it is parsed and then converted to the HCL color space. Uses d3-color’s hcl function.

# Back to Top

# Event Functions
# Functions for processing input event data. These functions are only legal in expressions evaluated in response to an event (for example a signal event handler). Invoking these functions elsewhere can result in errors.

# # item()
# Returns the current scenegraph item that is the target of the event.

# # group([name])
# Returns the scenegraph group mark item in which the current event has occurred. If no arguments are provided, the immediate parent group is returned. If a group name is provided, the matching ancestor group item is returned.

# # xy([item])
# Returns the x- and y-coordinates for the current event as a two-element array. If no arguments are provided, the top-level coordinate space of the view is used. If a scenegraph item (or string group name) is provided, the coordinate space of the group item is used.

# # x([item])
# Returns the x coordinate for the current event. If no arguments are provided, the top-level coordinate space of the view is used. If a scenegraph item (or string group name) is provided, the coordinate space of the group item is used.

# # y([item])
# Returns the y coordinate for the current event. If no arguments are provided, the top-level coordinate space of the view is used. If a scenegraph item (or string group name) is provided, the coordinate space of the group item is used.

# # pinchDistance(event)
# Returns the pixel distance between the first two touch points of a multi-touch event.

# # pinchAngle(event)
# Returns the angle of the line connecting the first two touch points of a multi-touch event.

# # inScope(item)
# Returns true if the given scenegraph item is a descendant of the group mark in which the event handler was defined, false otherwise.

# Back to Top

# Data Functions
# Functions for accessing Vega data sets.

# # data(name)
# Returns the array of data objects for the Vega data set with the given name. If the data set is not found, returns an empty array.

# # indata(name, field, value)
# Tests if the data set with a given name contains a datum with a field value that matches the input value. For example: indata('table', 'category', value).

# Back to Top

# Scale and Projection Functions
# Functions for working with Vega scale transforms and cartographic projections.

# # scale(name, value[, group])
# Applies the named scale transform (or projection) to the specified value. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale or projection.

# # invert(name, value[, group])
# Inverts the named scale transform (or projection) for the specified value. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale or projection.

# # copy(name[, group])
# Returns a copy (a new cloned instance) of the named scale transform of projection, or undefined if no scale or projection is found. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale or projection.

# # domain(name[, group])
# Returns the scale domain array for the named scale transform, or an empty array if the scale is not found. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale.

# # range(name[, group])
# Returns the scale range array for the named scale transform, or an empty array if the scale is not found. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale.

# # bandwidth(name[, group])
# Returns the current band width for the named band scale transform, or zero if the scale is not found or is not a band scale. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale.

# # bandspace(count[, paddingInner, paddingOuter])
# Returns the number of steps needed within a band scale, based on the count of domain elements and the inner and outer padding values. While normally calculated within the scale itself, this function can be helpful for determining the size of a chart’s layout.

# # gradient(scale, p0, p1[, count])
# Returns a linear color gradient for the scale (whose range must be a continuous color scheme) and starting and ending points p0 and p1, each an [x, y] array. The points p0 and p1 should be expressed in normalized coordinates in the domain [0, 1], relative to the bounds of the item being colored. If unspecified, p0 defaults to [0, 0] and p1 defaults to [1, 0], for a horizontal gradient that spans the full bounds of an item. The optional count argument indicates a desired target number of sample points to take from the color scale.

# # panLinear(domain, delta)
# Given a linear scale domain array with numeric or datetime values, returns a new two-element domain array that is the result of panning the domain by a fractional delta. The delta value represents fractional units of the scale range; for example, 0.5 indicates panning the scale domain to the right by half the scale range.

# # panLog(domain, delta)
# Given a log scale domain array with numeric or datetime values, returns a new two-element domain array that is the result of panning the domain by a fractional delta. The delta value represents fractional units of the scale range; for example, 0.5 indicates panning the scale domain to the right by half the scale range.

# # panPow(domain, delta, exponent)
# Given a power scale domain array with numeric or datetime values and the given exponent, returns a new two-element domain array that is the result of panning the domain by a fractional delta. The delta value represents fractional units of the scale range; for example, 0.5 indicates panning the scale domain to the right by half the scale range.

# # panSymlog(domain, delta, constant)
# Given a symmetric log scale domain array with numeric or datetime values parameterized by the given constant, returns a new two-element domain array that is the result of panning the domain by a fractional delta. The delta value represents fractional units of the scale range; for example, 0.5 indicates panning the scale domain to the right by half the scale range.

# # zoomLinear(domain, anchor, scaleFactor)
# Given a linear scale domain array with numeric or datetime values, returns a new two-element domain array that is the result of zooming the domain by a scaleFactor, centered at the provided fractional anchor. The anchor value represents the zoom position in terms of fractional units of the scale range; for example, 0.5 indicates a zoom centered on the mid-point of the scale range.

# # zoomLog(domain, anchor, scaleFactor)
# Given a log scale domain array with numeric or datetime values, returns a new two-element domain array that is the result of zooming the domain by a scaleFactor, centered at the provided fractional anchor. The anchor value represents the zoom position in terms of fractional units of the scale range; for example, 0.5 indicates a zoom centered on the mid-point of the scale range.

# # zoomPow(domain, anchor, scaleFactor, exponent)
# Given a power scale domain array with numeric or datetime values and the given exponent, returns a new two-element domain array that is the result of zooming the domain by a scaleFactor, centered at the provided fractional anchor. The anchor value represents the zoom position in terms of fractional units of the scale range; for example, 0.5 indicates a zoom centered on the mid-point of the scale range.

# # zoomSymlog(domain, anchor, scaleFactor, constant)
# Given a symmetric log scale domain array with numeric or datetime values parameterized by the given constant, returns a new two-element domain array that is the result of zooming the domain by a scaleFactor, centered at the provided fractional anchor. The anchor value represents the zoom position in terms of fractional units of the scale range; for example, 0.5 indicates a zoom centered on the mid-point of the scale range.

# Back to Top

# Geographic Functions
# Functions for analyzing geographic regions represented as GeoJSON features.

# # geoArea(projection, feature[, group])
# Returns the projected planar area (typically in square pixels) of a GeoJSON feature according to the named projection. If the projection argument is null, computes the spherical area in steradians using unprojected longitude, latitude coordinates. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the projection. Uses d3-geo’s geoArea and path.area methods.

# # geoBounds(projection, feature[, group])
# Returns the projected planar bounding box (typically in pixels) for the specified GeoJSON feature, according to the named projection. The bounding box is represented by a two-dimensional array: [[x₀, y₀], [x₁, y₁]], where x₀ is the minimum x-coordinate, y₀ is the minimum y-coordinate, x₁ is the maximum x-coordinate, and y₁ is the maximum y-coordinate. If the projection argument is null, computes the spherical bounding box using unprojected longitude, latitude coordinates. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the projection. Uses d3-geo’s geoBounds and path.bounds methods.

# # geoCentroid(projection, feature[, group])
# Returns the projected planar centroid (typically in pixels) for the specified GeoJSON feature, according to the named projection. If the projection argument is null, computes the spherical centroid using unprojected longitude, latitude coordinates. The optional group argument takes a scenegraph group mark item to indicate the specific scope in which to look up the projection. Uses d3-geo’s geoCentroid and path.centroid methods.

# Back to Top

# Tree (Hierarchy) Functions
# Functions for processing hierarchy data sets constructed with the stratify or nest transforms.

# # treePath(name, source, target)
# For the hierarchy data set with the given name, returns the shortest path through from the source node id to the target node id. The path starts at the source node, ascends to the least common ancestor of the source node and the target node, and then descends to the target node.

# # treeAncestors(name, node)
# For the hierarchy data set with the given name, returns the array of ancestors nodes, starting with the input node, then followed by each parent up to the root.

# Back to Top

# Browser Functions
# Functions for accessing web browser facilities.

# # containerSize()
# Returns the current CSS box size ([el.clientWidth, el.clientHeight]) of the parent DOM element that contains the Vega view. If there is no container element, returns [undefined, undefined].

# # screen()
# Returns the window.screen object, or {} if Vega is not running in a browser environment.

# # windowSize()
# Returns the current window size ([window.innerWidth, window.innerHeight]) or [undefined, undefined] if Vega is not running in a browser environment.

# Back to Top

# Logging Functions
# Logging functions for writing messages to the console. These can be helpful when debugging expressions.

# # warn(value1[, value2, …])
# Logs a warning message and returns the last argument. For the message to appear in the console, the visualization view must have the appropriate logging level set.

# # info(value1[, value2, …])
# Logs an informative message and returns the last argument. For the message to appear in the console, the visualization view must have the appropriate logging level set.

# # debug(value1[, value2, …])
# Logs a debugging message and returns the last argument. For the message to appear in the console, the visualization view must have the appropriate logging level set.


# From https://vega.github.io/vega/docs/expressions/
VEGAJS_NAMESPACE = {
    # Constants
    "NaN": math.nan,
    "E": math.e,
    "LN2": math.log(2),
    "LN10": math.log(10),
    "LOG2E": math.log2(math.e),
    "LOG10E": math.log10(math.e),
    "MAX_VALUE": sys.float_info.max,
    "MIN_VALUE": sys.float_info.min,
    "PI": math.pi,
    "SQRT1_2": math.sqrt(0.5),
    "SQRT2": math.sqrt(2),

    # Type Checking
    "isArray": isArray,
    "isBoolean": isBoolean,
    "isDate": isDate,
    "isNumber": isNumber,
    "isObject": isObject,
    "isRegExp": isRegExp,
    "isString": isString,

    # Type Coercion
    "toBoolean": toBoolean,
    "toDate": toDate,
    "toNumber": toNumber,
    "toString": toString,

    # Control Flow Functions
    "if": lambda test, if_value, else_value: if_value if test else else_value,

    # Math Functions
    "isNan": math.isnan,
    "isFinite": math.isfinite,
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "clamp": lambda val, low, hi: max(min(val, hi), low),
    "cos": math.cos,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "max": max,
    "min": min,
    "pow": math.pow,
    "random": random.random,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,

    # Date/Time Functions
    "now": now,
    "datetime": datetime,
    "date": date,
    "day": day,
    "year": year,
    "quarter": quarter,
    "month": month,
    "hours": hours,
    "minutes": minutes,
    "seconds": seconds,
    "milliseconds": milliseconds,
    "time": time,
    "timezoneoffset": timezoneoffset,
    "utc": utc,
    # TODOs:
    # Remaining Date/Time Functions
    # Array Functions
    # String Functions
    # Object Functions
    # Formatting Functions
    # RegExp Functions
}
