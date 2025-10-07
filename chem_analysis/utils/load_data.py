import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """ sorts in human order """
    return [atoi(c) for c in re.split(r'(\d+)', text)][1]


def get_files(path, pattern: str = "*.csv", sort_fun=natural_keys) -> list[str]:
    """ Returns a list of file names. """
    import glob
    import os

    # Find all json files
    file_list = []
    os.chdir(path)
    for files in glob.glob(pattern):
        file_list.append(files)  # filename with extension

    file_list.sort(key=sort_fun)
    return file_list