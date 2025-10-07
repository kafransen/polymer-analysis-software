import os
import glob
import datetime


def check_if_recent(folder: str, pattern: str = "temp*") -> int:
    os.chdir(folder)
    files = glob.glob(pattern)
    now = datetime.datetime.now()
    times = [os.path.getmtime(file) - now for file in files]

    for file, time in zip(files, times):
        if time < 1:
            return file
    else:
        return file