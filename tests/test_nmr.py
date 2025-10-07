def main():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\New folder\DW2-3_1\DW2-3\1")
    nmr = NMR.from_bruker(path)

    fig = go.Figure(
        go.Scatter(x=nmr.FID_time_axis, y=nmr.FID_real)
    )
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()