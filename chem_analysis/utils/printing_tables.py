from __future__ import annotations

from tabulate import tabulate

from chem_analysis.config import global_config


def apply_sig_figs(number: float | int | None, sig_digit: int = 3) -> int | float | None:
    """ significant figures
    Given a number return a string rounded to the desired significant digits.
    Parameters
    ----------
    number: float, int
        number you want to reduce significant figures on
    sig_digit: int
        significant digits
    Returns
    -------
    number: int, float
    """
    if isinstance(number, float):
        return float('{:0.{}}'.format(number, sig_digit))
    elif isinstance(number, int):
        return int(float('{:0.{}}'.format(float(number), sig_digit)))
    elif number is None:
        return None
    else:
        raise TypeError(f"'sig_figs' only accepts int or float. Given: {number} (type: {type(number)}")


class StatsTable:

    def __init__(self, rows: list[list], headers: list[str]):
        self.rows = rows
        self.headers = headers

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return f"rows: {len(self.rows)}, cols: {len(self.headers)}"

    def join(self, table: StatsTable, include_empty: bool = True):
        headers = self.headers + table.headers
        # remove duplicates while maintaining order
        seen = set()
        self.headers = [x for x in headers if x not in seen and not seen.add(x)]

        if table.rows:
            self.rows += table.rows
        else:
            if include_empty:
                self.rows += [[None] * len(self.headers)]

    def to_str(self, sig_figs: int = global_config.sig_fig, **kwargs):
        if "tablefmt" not in kwargs:
            kwargs["tablefmt"] = global_config.table_format
        rows = process_rows_to_str(self.rows, sig_figs)
        return tabulate(rows, self.headers, **kwargs)

    def to_csv_str(self, with_headers: bool = True, sig_figs: int = global_config.sig_fig) -> str:
        rows = process_rows_to_str(self.rows, sig_figs)
        if with_headers:
            inner_strings = [",".join(self.headers)]
        else:
            inner_strings = []

        inner_strings += [",".join(map(str, row)) for row in rows]
        return "\n".join(inner_strings)

    @classmethod
    def from_dict(cls, dict_: dict) -> StatsTable:
        headers = list(dict_.keys())
        return StatsTable(rows=[values_from_dict(dict_, headers, 0)], headers=["peak"] + headers)

    @classmethod
    def from_list_dicts(cls, list_: list[dict]) -> StatsTable:
        headers = get_headers_from_list_dicts(list_)
        return StatsTable(rows=values_from_list_of_dict(list_, headers), headers=["peak"] + headers)


def process_rows_to_str(rows: list[list], sig_figs: int) -> list[list]:
    rows_ = []
    for row in rows:
        row_ = []
        for v in row:
            row_.append(convert_to_str(v, sig_figs))
        rows_.append(row_)

    return rows_


def convert_to_str(value, sig_figs: int):
    if isinstance(value, float) or isinstance(value, int):
        value = apply_sig_figs(value, sig_figs)

    return value


def get_headers_from_list_dicts(list_) -> list[str]:
    if len(list_) == 0:
        return []

    headers = list(list_[0].keys())
    if len(list_) == 0:
        return headers

    for dict_ in list_[1:]:
        keys = dict_.keys()
        for k in keys:
            if k not in headers:
                headers.append(k)

    return headers


def values_from_dict(dict_: dict, headers: list[str], index: int) -> list:
    values = [index]
    for header in headers:
        values.append(dict_.get(header))

    return values


def values_from_list_of_dict(list_: list[dict], headers: list[str]) -> list:
    rows = []
    for i, dict_ in enumerate(list_):
        rows.append(values_from_dict(dict_, headers, i))

    return rows
