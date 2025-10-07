
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def func(t, x_0: float, delta: float, T: float, rad: float):
    return x_0 * (1 + delta * np.sin((2 * np.pi * t) / T + rad))


def main():
    t_total = 75
    n = t_total * 6
    t = np.linspace(0, t_total, n)  # minutes

    # res_time = func(t, 7, 0.7143, 55.5, np.pi)
    # light = func(t, 0.425, 0.8824, 222, np.pi)
    # cat_ratio = func(t, 8.75E-5, 0.7143, 111, 0)
    # temp = func(t, 303.15, 0.066, 444, 0) - 273

    res_time = func(t, 7.957,	0.4617,	154.7,	-0.262)
    light = func(t, 0.145,	0.1833,	205.5,	-2.391)
    cat_ratio = func(t, 5.48E-05,	0.3687,	45.7,	-1.243)
    temp = func(t, 308.98,	0.0411,	164.3,	2.901) - 273

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=['res_time', 'light', 'cat_ratio', 'temp'])
    fig.add_trace(go.Scatter(x=t, y=res_time), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=light), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=cat_ratio), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=temp), row=4, col=1)
    fig.layout.showlegend = False
    fig.write_html("temp0.html", auto_open=True)


class DynamicProfile:
    keys = ("res_time", "light", "ratio", "temp", "res_time_eff", "light_eff", "ratio_eff", "temp_eff")
    def __init__(self, profile: np.ndarray):
        self.profile = profile

    @property
    def time(self) -> np.ndarray:
        return self.profile[:, 0] * 60  # min to sec conversion

    @property
    def res_time(self) -> np.ndarray:
        return self.profile[:, 1]

    @property
    def light(self):
        return self.profile[:, 2]

    @property
    def ratio(self):
        return self.profile[:, 3]

    @property
    def temp(self):
        return self.profile[:, 4] - 273.15

    @property
    def res_time_eff(self) -> np.ndarray:
        return self.profile[:, 5]

    @property
    def light_eff(self):
        return self.profile[:, 6]

    @property
    def ratio_eff(self):
        return self.profile[:, 7]

    @property
    def temp_eff(self):
        return self.profile[:, 8] - 273.15

    def get_values(self, t: float | np.ndarray) -> np.ndarray:
        data = []
        for k in self.keys:
            data.append(np.interp(t, self.time, getattr(self, k)))

        return np.array(data)


def load_profiles() -> DynamicProfile:
    # t, temp, light, flow_rate_mon, flow_rate_cat, flow_rate_dmso
    data = np.loadtxt(r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-7\effective.csv", delimiter=",")
    return DynamicProfile(data)


def main2():
    profiles = load_profiles()
    ir_times = np.loadtxt(r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-10\IR_times.csv", delimiter=",")

    data = profiles.get_values(ir_times[:,0])
    fig = go.Figure()

    scatter = go.Scatter3d(
        x=data[4],
        y=data[5],
        z=ir_times[:,1],
        mode='markers',
        marker=dict(
            size=6,
            color=data[6],  # Use the color values for the fourth dimension
            colorscale='Viridis',  # You can choose other color scales
            opacity=0.8
        )
    )
    fig.update_layout(
    scene=dict(
        xaxis_title='res_time',
        yaxis_title='light',
        zaxis_title='conversion'
    ),
    title='Color is CTA ratio'
)
    fig.add_trace(scatter)

    from plotly_gif import three_d_scatter_rotate, GIF
    gif = GIF()
    three_d_scatter_rotate(gif, fig, auto_create=False)
    gif.create_gif(length=10_000)
    fig.show()


if __name__ == "__main__":
    main()
