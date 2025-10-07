
import numpy as np
import plotly.graph_objs as go


def main():
    path = r"C:\Users\nicep\Desktop\DW2-flow_rate\DW2-flow_rate\_282\data.1d"
    dtype_ = np.float64

    with open(path, mode='rb') as f:
        d = f.read()
        a = np.frombuffer(d, dtype=dtype_)


    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data))
    print("hi")


if __name__ == "__main__":
    main()
