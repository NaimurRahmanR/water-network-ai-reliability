# AQUA-VERA Dataset Downloader v3

The previous downloader's `--full-battledim` label was misleading: it selected the CSV research core, not every file published in the official BattLeDIM record. v3 fixes that.

## Option 1 — complete BattLeDIM (~550.5 MB)
Double-click:

`download_windows_FULL_BATTLEDIM.bat`

This downloads all 17 files from Zenodo record 4017659, including both XLSX files and `L-TOWN_Real.inp`, with official MD5 verification.

## Option 2 — everything (~14.5 GB before extraction)
Double-click:

`download_windows_EVERYTHING_14GB.bat`

This downloads:
- complete BattLeDIM (~550.5 MB)
- EPA WNTR Net3
- complete LeakDB.zip (~13.9 GB)

LeakDB is checksum-verified against MD5 `d607929ef7caea432e663aee8071f2d7`.

## Existing data
Do not delete `C:\Users\<you>\Downloads\AQUA_VERA_data`. Valid files are skipped. `.part` files are resumed.

Official sources:
- BattLeDIM DOI: 10.5281/zenodo.4017659
- LeakDB DOI: 10.5281/zenodo.13985057
- WNTR Net3: USEPA/WNTR GitHub repository
