
import chem_analysis as ca


def main():
    path = r"C:\Users\nicep\Desktop\DW2_11_p2\1"
    nmr = ca.nmr.NMRSignal.from_bruker(path)

    fig = ca.plotting.signal(nmr)
    fig.show()


if __name__ == "__main__":
    main()
