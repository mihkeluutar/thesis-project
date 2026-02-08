"""
Read and inspect the first ERA5 file from data/copernicus-era5-taltech.
Files may be NetCDF or ZIP containing GRIB (CDS sometimes returns GRIB).
Run from thesis-project root: python scripts/read_era5_sample.py
Requires: pip install xarray netcdf4 cfgrib
"""

import zipfile
import tempfile
from pathlib import Path

try:
    import xarray as xr
except ImportError:
    print("Install with: pip install xarray netcdf4 cfgrib")
    raise

# Paths relative to thesis-project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "copernicus-era5-taltech"

# First surface file (earliest by name)
surface_files = sorted(DATA_DIR.glob("taltech_era5land_surface_*.*"))
# Include both .nc and any extension (in case they're actually .zip or .grib)
if not surface_files:
    surface_files = sorted(DATA_DIR.glob("taltech_era5land_surface_*.nc"))
if not surface_files:
    print("No taltech_era5land_surface_* files found in", DATA_DIR)
    raise SystemExit(1)

first_surface = surface_files[0]
print("Opening:", first_surface.name)
print()

ds = None

# Check if file is actually a ZIP containing GRIB (CDS often returns this)
if zipfile.is_zipfile(first_surface):
    print("(File is ZIP with GRIB inside — extracting and reading with cfgrib)")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(first_surface, "r") as z:
            names = z.namelist()
            grib_name = next((n for n in names if "grib" in n.lower() or n.endswith(".grb")), names[0])
            path = Path(z.extract(grib_name, path=tmp))
        ds = xr.open_dataset(path, engine="cfgrib")
else:
    # Real NetCDF
    try:
        ds = xr.open_dataset(first_surface, engine="netcdf4")
    except (ValueError, OSError):
        ds = xr.open_dataset(first_surface, engine="scipy")

if ds is None:
    print("Could not open file.")
    raise SystemExit(1)

print("=== Dataset structure ===")
print(ds)
print()

print("=== Coordinates ===")
for name, coord in ds.coords.items():
    print(f"  {name}: dims={coord.dims}, shape={coord.shape}")
print()

print("=== Data variables ===")
for name, da in ds.data_vars.items():
    print(f"  {name}: dims={da.dims}, shape={da.shape}")
    if hasattr(da, "attrs") and da.attrs.get("units"):
        print(f"    units: {da.attrs.get('units')}")
print()

print("=== Small sample (first 3 times, single point) ===")
if "time" in ds.dims or "valid_time" in ds.dims:
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    ntime = min(3, ds.sizes[time_dim])
    sample = ds.isel({time_dim: slice(0, ntime)})
    for d in ("latitude", "longitude", "lat", "lon"):
        if d in sample.dims and sample.sizes.get(d, 0) == 1:
            sample = sample.isel({d: 0})
    print(sample)
else:
    print(ds)

ds.close()
print()
print("Done. File is readable.")
