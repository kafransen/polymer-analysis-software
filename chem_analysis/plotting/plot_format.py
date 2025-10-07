import re

import numpy as np


def bold_in_html(text: str):
    return f"<b>{text}</b>"


hex_options = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F')


def get_plot_color(num_colors: int = 1) -> list[str]:
    """Gets color for 2D plots."""
    color_list = [
        'rgb(10,36,204)',  # blue
        'rgb(172,24,25)',  # red
        'rgb(6,127,16)',  # green
        'rgb(251,118,35)',  # orange
        'rgb(145,0,184)',  # purple
        'rgb(255,192,0)'  # yellow
    ]
    if num_colors <= 1:
        return [color_list[0]]
    if num_colors <= len(color_list):
        return color_list[:num_colors]
    else:
        num_colors_extra = num_colors - len(color_list)
        for i in range(num_colors_extra):
            color = ["#" + ''.join([np.random.choice(hex_options) for _ in range(6)])]
            color = [col.lstrip('#') for col in color]
            color = ["rgb" + str(tuple(int(col[i:i + 2], 16) for i in (0, 2, 4))) for col in color]
            color_list = color_list + color

        return color_list


def get_similar_color(color_in: str, num_colors: int, mode: str = "dark") -> list[str]:
    rgb = re.findall("[0-9]{1,3}", color_in)
    rgb = [int(i) for i in rgb]
    if mode == "dark":
        change_rgb = [i > 120 for i in rgb]
        jump_amount = [-int((i - 80) / num_colors) for i in rgb]
        jump_amount = [v if i else 0 for i, v in zip(change_rgb, jump_amount)]

    elif mode == "light":
        jump_amount = [int(100 / num_colors) if i < 100 else int((245-i)/num_colors) for i in rgb]

    else:
        raise ValueError(f"Invalid 'mode'; only 'light' or 'dark'. (mode: {mode})")

    colors = []
    for i in range(num_colors):
        r = rgb[0] + jump_amount[0] * (i+1)
        g = rgb[1] + jump_amount[1] * (i+1)
        b = rgb[2] + jump_amount[2] * (i+1)
        colors.append(f"rgb({r},{g},{b})")

    return colors
