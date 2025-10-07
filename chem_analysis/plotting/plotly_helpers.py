import os
from typing import Sequence

import plotly.graph_objs as go


def merge_html_figs(figs: Sequence[go.Figure], filename: str = "merge.html", auto_open: bool = True):
    """
    Merges plotly figures into single html

    Parameters
    ----------
    figs: list[go.Figure, str]
        list of figures to append together
    filename:str
        file name
    auto_open: bool
        open html in browser after creating
    """
    if filename[-5:] != ".html":
        filename += ".html"

    with open(filename, 'w', encoding="UTF-8") as file:
        file.write(f"<html><head><title>{filename[:-5]}</title><h1>{filename[:-5]}</h1></head><body>" + "\n")
        for fig in figs:
            if isinstance(fig, str):
                file.write(fig)
                continue

            # inner_html = fig.to_html(include_plotlyjs="cdn").split('<body>')[1].split('</body>')[0]
            inner_html = fig.to_html(full_html=False)
            file.write(inner_html)

        file.write("</body></html>" + "\n")

    if auto_open:
        os.system(fr"start {filename}")
