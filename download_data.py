"""Download the official UCI CCPP workbook into data/.

The analysis uses only the first worksheet, as required by Homework 2.
"""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io
import shutil

URL = "https://archive.ics.uci.edu/static/public/294/combined%2Bcycle%2Bpower%2Bplant.zip"
OUT = Path(__file__).resolve().parent / "data" / "Folds5x2_pp.xlsx"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    payload = urlopen(URL).read()
    with ZipFile(io.BytesIO(payload)) as archive:
        candidates = [n for n in archive.namelist() if n.endswith("Folds5x2_pp.xlsx")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one workbook, found: {candidates}")
        with archive.open(candidates[0]) as src, OUT.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

