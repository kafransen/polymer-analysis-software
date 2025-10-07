import pathlib
import os


def rename_spectra(path: pathlib.Path):
    folders = tuple(os.scandir(path))
    for folder in folders:
        modified_lines = []
        with open(pathlib.Path(folder.path) / "acqu.par", 'r') as file:
            for line in file:
                if line.startswith("Sample"):
                    # Replace the line that starts with "Sample ="
                    modified_lines.append(line.replace("DW2-5-1", f"DW2-5-1_{folder.name[1:]}"))
                else:
                    modified_lines.append(line)

        # Open the file in write mode and write the modified lines
        with open(pathlib.Path(folder.path) / "acqu.par", 'w') as file:
            file.writelines(modified_lines)


def main():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\DW2-flow_rate")
    rename_spectra(path)


if __name__ == "__main__":
    main()
