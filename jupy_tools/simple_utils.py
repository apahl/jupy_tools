#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
This is the simplified version without the RDKit functions.
"""

import os
import os.path as op
from pathlib import Path
import datetime
import platform
from glob import glob
import subprocess
import tempfile
import uuid
import signal
import time
from contextlib import contextmanager

from typing import Any, Callable, List, Set, Tuple, Union

import pandas as pd
from pandas.core.frame import DataFrame

import numpy as np

from multiprocessing import Pool


INTERACTIVE = True
MIN_NUM_RECS_PROGRESS = 500
INFO_WIDTH = 35


def is_interactive_ipython():
    try:
        get_ipython()  # type: ignore
        ipy = True
    except NameError:
        ipy = False
    return ipy


NOTEBOOK = is_interactive_ipython()
if NOTEBOOK:
    try:
        from tqdm.notebook import tqdm

        tqdm.pandas()
        TQDM = True
    except ImportError:
        TQDM = False
else:
    try:
        from tqdm import tqdm

        tqdm.pandas()
        TQDM = True
    except ImportError:
        TQDM = False


class MeasureRuntime:
    """Measure the elapsed time between two points in the code."""

    def __init__(self):
        self.start = time.time()

    def elapsed(self, show=True, msg="Runtime"):
        """Print (show=True) or return (show=False, in seconds) a timestamp for the runtime since."""
        run_time = time.time() - self.start
        if show:
            time_unit = "s"
            if run_time > 120:
                run_time /= 60
                time_unit = "min"
                if run_time > 300:
                    run_time /= 60
                    time_unit = "h"
                    if run_time > 96:
                        run_time /= 24
                        time_unit = "d"
            print(f"{msg}: {run_time:.1f} {time_unit}")
        else:
            return run_time


def timestamp(show=True):
    """Print (show=True) or return (show=False) a timestamp string."""
    info_string = (
        f'{time.strftime("%d-%b-%Y %H:%M:%S")} ({os.getlogin()} on {platform.system()})'
    )
    if show:
        print("Timestamp:", info_string)
    else:
        return info_string


class Results:
    """
    A utility class for collecting, displaying, and formatting result entries.

    Features:
    - Stores results as a list of (name, value) tuples.
    - Provides pretty-printing in text and HTML (for Jupyter notebooks).
    - Supports adding, removing, and temporarily displaying results.
    - Can clear all results or remove the last n entries.

    Methods:
        add(*res, show=True): Add one or more result tuples or section headers.
        remove(n): Remove the last n entries.
        tmp(*res): Temporarily add and show entries, then remove them.
        clear(): Remove all entries.
        show(idx=0): Return a formatted string of results from idx onward.
        to_html(): Return a pandas DataFrame of results (for Jupyter display).
    """

    def __init__(self, headers=["Result", "Value"]):
        self.headers = headers
        self.list = []

    def show(self, idx: int = 0):
        col0_max = max([len(x[0]) for x in self.list[idx:]])
        col1_max = max([len(x[1]) for x in self.list[idx:]])
        if col0_max < len(self.headers[0]):
            col0_max = len(self.headers[0])
        if col1_max < len(self.headers[1]):
            col1_max = len(self.headers[1])
        sep = "―" * (col0_max + col1_max + 2)
        out = [f"{self.headers[0]:{col0_max}s}  {self.headers[1]:>{col1_max}s}"]
        out.append(sep)
        for rn, r in enumerate(self.list[idx:]):
            if rn == 0 and r[0] == " ":
                continue  # skip empty line if it is the first line to show (can happen for `idx` > 0)
            out.append(f"{r[0]:{col0_max}s}  {r[1]:>{col1_max}s}")
        return "\n".join(out)

    def __str__(self):
        return self.show()

    def __repr__(self):
        return self.show()

    def to_html(self):
        if NOTEBOOK:
            return pd.DataFrame.from_records(self.list, columns=self.headers)
        else:
            return ""

    def add(self, *res, show=True):
        """Add one or more result tuples to the instance.
        Show the added entries."""
        idx = len(self.list)
        for r in res:
            if isinstance(r, str):
                if len(self.list) > 0:
                    self.list.append((" ", " "))
                self.list.append((r, " "))
                continue
            r = list(r)
            if len(r) < 2:
                r.append(" ")
                if len(self.list) > 0:
                    self.list.append((" ", " "))
            else:
                r[0] = "• " + r[0]
                if isinstance(r[1], float):
                    r[1] = f"{r[1]:.3f}"
                else:
                    r[1] = str(r[1])
            self.list.append(r)
        if show:
            print(self.show(idx))

    def remove(self, n):
        """Remove the n last entries."""
        self.list = self.list[:-n]

    def tmp(self, *res):
        """Temporarily add an entry.
        During development. Add, show, delete."""
        self.add(res)
        self.remove(len(res))

    def clear(self):
        self.list = []


def lp(obj, label: str = None, lpad=INFO_WIDTH, rpad=7):
    """log-printing for different kind of objects"""
    if label is not None:
        label_str = label
    if isinstance(obj, str):
        if label is None:
            label_str = "String"
        print(f"{label_str:{lpad}s}: {obj:>{rpad}s}")
        return

    try:
        shape = obj.shape
        if label is None:
            label_str = "Shape"
        else:
            label_str = f"Shape {label}"
        key_str = ""
        has_nan_str = ""
        try:
            keys = list(obj.columns)
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
            num_nan_cols = ((~obj.notnull()).sum() > 0).sum()
            if num_nan_cols > 0:  # DF has nans
                has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        except AttributeError:
            pass
        print(
            f"{label_str:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        shape = obj.data.shape
        if label is None:
            label_str = "Shape"
        else:
            label_str = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.data.columns)
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.data.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label_str:{lpad}s}:   {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        fval = float(obj)
        if label is None:
            label_str = "Number"
        if fval == obj:
            print(f"{label_str:{lpad}s}:   {int(obj):{rpad}d}")
        else:
            print(f"{label_str:{lpad}s}:   {obj:{rpad+6}.5f}")
        return
    except (ValueError, TypeError):
        # print("Exception")
        pass

    try:
        length = len(obj)
        if label is None:
            label_str = "Length"
        else:
            label_str = f"Length {label}"
        print(f"{label_str:{lpad}s}:   {length:{rpad}d}")
        return
    except (TypeError, AttributeError):
        pass

    if label is None:
        label_str = "Object"
    print(f"{label_str:{lpad}s}:   {obj}")


def info(df: pd.DataFrame, fn: str = "Shape", what: str = ""):
    """Print information about the result from a function,
    when INTERACTIVE is True.

    Parameters:
    ===========
    df: the result DataFrame
    fn: the name of the function
    what: the result of the function"""
    if not isinstance(df, pd.DataFrame):
        lp(df, fn)
        return
    if len(what) > 0:
        what = f"{what} "
    shape = df.shape
    keys = ""
    if shape[1] < 10:
        keys = ", ".join(df.keys())
        if len(keys) < 80:
            keys = f"( {keys} )"
    print(f"{fn:{INFO_WIDTH}s}: [ {shape[0]:7d} / {shape[1]:3d} ] {what}{keys}")


def get_value(str_val):
    """convert a string into float or int, if possible."""
    if not str_val:
        return np.nan
    try:
        if "." in str_val:
            val = float(str_val)
        else:
            val = int(str_val)
    except ValueError:
        val = str_val
    return val


def count_nans(df: pd.DataFrame, columns: Union[str, List[str], None] = None) -> int:
    """Count rows containing NANs in the `column`.
    When no column is given, count all NANs."""
    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
    column_list = []
    nan_counts = []
    for col in columns:
        column_list.append(col)
        nan_counts.append(df[col].isna().sum())
    if INTERACTIVE and len(columns) == 1:
        fn = "count_nans"
        print(
            f"{fn:25s}: [ {nan_counts[0]:6d}       ] rows with NAN values in col `{columns[0]}`"
        )
    return pd.DataFrame({"Column": column_list, "NANs": nan_counts})


def remove_nans(
    df: pd.DataFrame, column: Union[str, List[str]], reset_index=True
) -> pd.DataFrame:
    """Remove rows containing NANs in the `column`.

    Parameters:
    ===========
    df: pd.DataFrame
        The DataFrame to be processed
    column: Union[str, List[str]]
        The column(s) in which the nans should be replaced.

    Returns: A new DataFrame without the rows containing NANs.
    """
    result = df.copy()
    if isinstance(column, str):
        column = [column]
    for col in column:
        result = result[result[col].notna()]
        if INTERACTIVE:
            info(
                result,
                f"remove_nans `{col[:INFO_WIDTH-14]}`",
                f"{len(df) - len(result):4d} rows removed.",
            )
    if reset_index:
        result = result.reset_index(drop=True)
    return result


def replace_nans(
    df: pd.DataFrame, columns: Union[str, List[str]], value: Any
) -> pd.DataFrame:
    """Replace fields containing NANs in the `column` with `value`.

    Parameters:
    ===========
    df: pd.DataFrame
        The DataFrame to be processed
    column: Union[str, List[str]]
        The column(s) in which the nans should be replaced.
    value: Any
        the value by which the nans should be replaced.

    Returns: A new DataFrame where the NAN fields have been replaced by `value`.
    """
    result = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        mask = result[col].isna()
        num_nans = mask.sum()
        if isinstance(value, str):
            result[col] = result[col].astype(str)
            result.loc[mask, col] = value
        else:
            result = result.fillna({col: value})
        if INTERACTIVE:
            info(
                result,
                f"replace_nans `{col[:INFO_WIDTH-15]}`",
                f"{num_nans:4d} values replaced.",
            )
    return result


def drop_cols(df: pd.DataFrame, cols: Union[str, List[str]]) -> pd.DataFrame:
    """Remove the column or the list of columns from the dataframe.
    Listed columns that are not available in the dataframe are simply ignored."""
    if not isinstance(cols, list):
        cols = [cols]
    shape1 = df.shape
    df = df.copy()
    cols_to_remove = set(cols).intersection(set(df.keys()))
    df = df.drop(cols_to_remove, axis=1)
    shape2 = df.shape
    if INTERACTIVE:
        info(df, "drop_cols", f"{shape1[1] - shape2[1]:2d} columns removed.")
    return df


def reorder_list(lst: List[Any], take: Union[List[Any], Any], front=True) -> List[Any]:
    """Reorder the given list `lst`, so that the elements in `take` are at the front (front=True)
    or at the end (front=False) of the list. The order of the elements in `take` will be preserved.
    If `take` contains elements that are not in `lst`, a ValueError will be raised.
    Returns: the reordered list."""
    if not isinstance(take, list):
        take = [take]
    for el in take:
        if el in lst:
            lst.remove(el)
        else:
            raise ValueError(f"Element `{el}` not in list")
    if front:
        result = take + lst
    else:
        result = lst + take
    return result


def bring_to_front(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """Bring the column(s) `columns` to the front of the DataFrame. `columns` can be a single column
    or a list of columns. The order of the columns in `columns` will be preserved. If `columns` contains
    names that are not present in the DataFrame, a ValueError will be raised."""
    if isinstance(columns, str):
        columns = [columns]
    cols = df.columns.tolist()
    for key in columns:
        if key in cols:
            cols.remove(key)
        else:
            raise ValueError(f"Column `{key}` not in DataFrame")
    cols = [key] + cols
    return df[cols]


def parallel_pandas(df: pd.DataFrame, func: Callable, workers=6) -> pd.DataFrame:
    """Concurrently apply the `func` to the DataFrame `df`.
    `workers` is the number of parallel threads.
    Currently, TQDM progress bars do not work with the parallel execution.

    Returns:
    ========
    A new Pandas DataFrame.

    Example:
    ========

    >>> def add_props(df):
    >>>     df["Mol"] = df["Smiles"].apply(u.smiles_to_mol)
    >>>     df["LogP"] = df["Mol"].apply(Desc.MolLogP)
    >>>     return df

    >>>     dfs = u.parallel_pandas(df, add_props)
    """
    df = df.copy()
    df_split = np.array_split(df, workers)
    pool = Pool(workers)
    # if TQDM:
    #     result = pd.concat(pool.map(func, df_split))
    # else:
    result = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return result


def read_tsv(
    input_tsv: str, sep="\t", encoding="utf-8", index_col=None
) -> pd.DataFrame:
    """Read a tsv file

    Parameters:
    ===========
    input_tsv: Input tsv file. Can also be pathlib.Path object.

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    if isinstance(input_tsv, str):
        input_tsv = input_tsv.replace("file://", "")
    p_input_tsv = Path(input_tsv)
    df = pd.read_csv(
        p_input_tsv, sep=sep, encoding=encoding, low_memory=False, index_col=index_col
    )
    if INTERACTIVE:
        time_stamp = datetime.datetime.fromtimestamp(
            p_input_tsv.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")
        info(df, f"read_tsv (mod.: {time_stamp})")
    return df


def read_chunked_tsv(pattern: str, sep="\t") -> pd.DataFrame:
    """
    Read a list of chunked CSV files into one concatenated DataFrame.

    Parameters
    ==========
    pattern: str
        A glob pattern for the chunked CSV files.
        Example: 'data/chunked/*.csv'
    sep: str
        The delimiter for the columns in the CSV files. Default: TAB.

    Returns: pd.DataFrame with the concatenated data.
    """
    chunks = []
    file_list = glob(pattern)
    for f in file_list:
        chunks.append(pd.read_csv(f, sep=sep))
    result = pd.concat(chunks)
    if INTERACTIVE:
        info(result, f"read_chunked_tsv ({len(file_list)})")
    return result


def write(fn, text):
    """Write text to a file."""
    with open(fn, "w") as f:
        f.write(text)


def write_tsv(df: pd.DataFrame, output_tsv: str, sep="\t"):
    """Write a tsv file, converting the RDKit molecule column to smiles.

    Parameters:
    ===========
    input_tsv: Input tsv file

    """
    # The Mol column can not be saved to TSV in a meaningfull way,
    # so we remove it, if it is present.
    if "Mol" in df.keys():
        df = df.drop("Mol", axis=1)
    df.to_csv(output_tsv, sep=sep, index=False)


def save_list(lst, fn="list.txt"):
    """Save list as text file."""
    with open(fn, "w") as f:
        for line in lst:
            f.write(f"{line}\n")


def load_list(fn="list.txt", as_type=str, skip_remarks=True, skip_empty=True):
    """Read the lines of a text file into a list.

    Parameters:
    ===========
    as_type: Convert the values in the file to the given format. (Default: str).
    skip_remarks: Skip lines starting with `#` (default: True).
    skip_empty: Skip empty lines. (Default: True).

    Returns:
    ========
    A list of values of the given type.
    """
    result = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if skip_empty and len(line) == 0:
                continue
            if skip_remarks and line.startswith("#"):
                continue
            result.append(as_type(line))
    return result


def open_in_localc(df: pd.DataFrame):
    """Open a Pandas DataFrame in LO Calc for visual inspection."""
    td = tempfile.gettempdir()
    tf = str(uuid.uuid4()).split("-")[0] + ".tsv"
    path = op.join(td, tf)
    write_tsv(df, path)
    subprocess.Popen(["localc", path])


def open_in_excel(df: pd.DataFrame):
    """Open a Pandas DataFrame in MS Excel for visual inspection."""
    td = tempfile.gettempdir()
    tf = str(uuid.uuid4()).split("-")[0] + ".tsv"
    path = op.join(td, tf)
    write_tsv(df, path)
    os.startfile(path)


def listify(s, sep=" ", as_int=True, strip=True, sort=False):
    """A helper func for the Jupyter Notebook,
    which generates a correctly formatted list out of pasted text.

    Parameters:
    ===========
    as_int: The function always attempts to convert the entries to numbers This option controls whether the numbers are converted to int (default: true) or float (false).
    sort: Sort the output list (default: False).
    """
    to_number = int if as_int else float
    result = []
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    lst = s.split(sep)
    for el in lst:
        if strip:
            el = el.strip()
        if len(el) == 0:
            continue
        try:
            el = to_number(el)
        except ValueError:
            pass
        result.append(el)
    return result


def id_filter(df, id_list, id_col, reset_index=True, sort_by_input=False):
    """Filter a dataframe by a list of IDs.
    If `sort_by_input` is True, the output is sorted by the input list."""
    if isinstance(id_list, str) or isinstance(id_list, int):
        id_list = [id_list]
    result = df[df[id_col].isin(id_list)]

    if reset_index:
        result = result.reset_index(drop=True)
    if sort_by_input:
        result["_sort"] = pd.Categorical(
            result[id_col], categories=id_list, ordered=True
        )
        result = result.sort_values("_sort")
        result = result.drop("_sort", axis=1)
    return result


def filter(
    df: pd.DataFrame, mask, reset_index=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filters a dataframe and returns the passing fraction and the failing fraction as
    two separate dataframes.

    Returns: passing and failing dataframe."""
    df_pass = df[mask].copy()
    df_fail = df[~df.index.isin(df_pass.index)].copy()
    if reset_index:
        df_pass = df_pass.reset_index(drop=True)
        df_fail = df_fail.reset_index(drop=True)
    if INTERACTIVE:
        info(df_pass, "filter: pass")
        info(df_fail, "filter: fail")
    return df_pass, df_fail


def inner_merge(
    df_left: pd.DataFrame, df_right: pd.DataFrame, on: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inner merge for two dataframes that also reports the entries from the left df that were not found in the right df.

    Returns: two dataframes, the merged dataframe and missing entries from left df."""
    df_result = df_left.merge(df_right, on=on, how="inner").reset_index(drop=True)
    df_result_on = df_result[on].unique()
    df_missing = df_left[~df_left[on].isin(df_result_on)].reset_index(drop=True)
    if INTERACTIVE:
        info(df_result, "merge_inner: result")
        info(df_missing, "merge_inner: missing")
    return df_result, df_missing


def drop_duplicates(
    df: pd.DataFrame, subset: Union[str, List[str]], reset_index=True
) -> Tuple[pd.DataFrame, List[Any]]:
    """Drops the duplicates from the given dataframe and returns it as well as the duplicates as list

    Returns: dataframe without duplicates and a list of the duplicates.."""
    df_pass = df.copy().drop_duplicates(subset=subset)
    if reset_index:
        df_pass = df_pass.reset_index(drop=True)
    if isinstance(subset, str):
        subset_list = [subset]
    tmp = df[subset_list].copy()
    tmp["CountXX"] = 1
    tmp = tmp.groupby(by=subset).count().reset_index()
    tmp = tmp[tmp["CountXX"] > 1].copy()
    dupl_list = tmp[subset].values.tolist()
    if INTERACTIVE:
        info(df_pass, "drop_dupl: result")
        info(dupl_list, "drop_dupl: dupl")
    return df_pass, dupl_list


def groupby(df_in, by=None, num_agg=["median", "mad", "count"], str_agg="unique"):
    """Other str_aggs: "first", "unique"."""

    def _concat(values):
        return "; ".join(str(x) for x in values)

    def _unique(values):
        return "; ".join(set(str(x) for x in values))

    if isinstance(num_agg, str):
        num_agg = [num_agg]
    df_keys = df_in.columns
    numeric_cols = list(df_in.select_dtypes(include=[np.number]).columns)
    str_cols = list(set(df_keys) - set(numeric_cols))
    # if by in numeric_cols:
    try:
        by_pos = numeric_cols.index(by)
        numeric_cols.pop(by_pos)
    except ValueError:
        pass
    try:
        by_pos = str_cols.index(by)
        str_cols.pop(by_pos)
    except ValueError:
        pass
    aggregation = {}
    for k in numeric_cols:
        aggregation[k] = num_agg
    if str_agg == "join":
        str_agg_method = _concat
    elif str_agg == "first":
        str_agg_method = "first"
    elif str_agg == "unique":
        str_agg_method = _unique
    for k in str_cols:
        aggregation[k] = str_agg_method
    df = df_in.groupby(by)
    df = df.agg(aggregation).reset_index()
    df_cols = [
        "_".join(col).strip("_").replace("_<lambda>", "").replace("__unique", "")
        for col in df.columns.values
    ]
    df.columns = df_cols
    if INTERACTIVE:
        info(df, "group_by")
    return df


def split_df_in_chunks(df: pd.DataFrame, num_chunks: int, base_name: str):
    """Splits the given DataFrame into `num_chunks` chunks and writes them to separate TSV files,
    using `base_name.`
    """
    if "." in base_name:
        pos = base_name.rfind(".")
        base_name = base_name[:pos]
    chunk_size = (len(df) // num_chunks) + 1
    ndigits = 2 if num_chunks > 9 else 1
    ctr = 0
    for i in range(0, df.shape[0], chunk_size):
        ctr += 1
        write_tsv(df[i : i + chunk_size], f"{base_name}_{ctr:{ndigits}d}.tsv")


# Timeout code is taken from José-Manuel Gally's NPFC project:
# https://github.com/mpimp-comas/npfc/blob/master/npfc/utils.py
# See also references cited there.
def raise_timeout(signum, frame):
    """Function to actually raise the TimeoutError when the time has come."""
    raise TimeoutError


@contextmanager
def timeout(time):
    """Context manager to raise a TimeoutError if the given time in seconds has passed.
    Example usage:
    >>> import time
    >>> timed_out = True
    >>> with timeout(5):
    >>>     time.sleep(6)  # put some actual code here
    >>>     timed_out = False
    >>> if timed_out:
    >>>     print("Timed out!")
    """
    # register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # schedule the signal to be sent after time
    signal.alarm(time)
    # run the code block within the with statement
    try:
        yield
    except TimeoutError:
        pass  # exit the with statement
    finally:
        # unregister the signal so it won't be triggered if the timeout is not reached
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


# Pandas extensions
def pandas_extensions():
    """Adds the following extensions to the Pandas DataFrame:
    - `iquery`: same as the DF `query`, but prints info about the shape of the result.
    - `ifilter`
    - `imerge`
    - `idrop_duplicates`


    See also doc for: `filter`, `inner_merge` from this module."""

    DataFrame.ifilter = filter
    DataFrame.imerge = inner_merge
    DataFrame.idrop_duplicates = drop_duplicates

    def inner_query(
        df: pd.DataFrame, query: str, reset_index=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Queries a dataframe using pandas `query` syntax
        and returns the passing fraction and the failing fraction as
        two separate dataframes.

        Returns: passing and failing dataframe."""
        df_pass = df.query(query).copy()
        df_fail = df[~df.index.isin(df_pass.index)].copy()
        if reset_index:
            df_pass = df_pass.reset_index(drop=True)
            df_fail = df_fail.reset_index(drop=True)
        if INTERACTIVE:
            info(df_pass, "query_pass")
            info(df_fail, "query_fail")
        return df_pass, df_fail

    DataFrame.iquery = inner_query
