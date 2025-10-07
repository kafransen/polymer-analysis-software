import pathlib

import chem_analysis as ca


def main():
    path = r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\IR_standards\PDMAA_DMSO_20_percent.csv"
    path = pathlib.Path(path)
    sig = ca.ir.IRSignal.from_file(path)

    sig.processor.add(
        ca.processing.baselines.AsymmetricLeastSquared(
            lambda_=1e6,
            weights=ca.processing.weigths.Spans(x_spans=[(400, 600), (1900, 2800)], invert=True)
        )
    )

    fig = ca.plotting.signal(sig)
    fig = ca.plotting.signal_raw(sig, fig=fig)
    fig = ca.plotting.baseline(sig, fig=fig)
    fig.show()

    for i in range(len(sig.x)):
        print(f"{sig.x[i]}, {sig.y[i]}")


if __name__ == "__main__":
    main()
