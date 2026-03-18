"""
Data Loading Module for ES Data Pipeline.
Handles loading of building energy data and weather data from various sources.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple, Union
import glob
import os

from .config import (
    DataStatus,
    COL_TIME,
    COL_HEAT,
    COL_TEMP_OUT,
    COL_WIND_SPEED,
    COL_SOLAR,
    COL_STATUS,
    WEATHER_COL_MAPPING
)

# Defaults (can be overridden)
DEFAULT_CAMPUS_DATA_DIR = Path("data/campus-data")
DEFAULT_WEATHER_DATA_PATH_KESK = Path("data/keskkonnaportaal/tallinn-harku_f_kliima_tund.csv")
DEFAULT_ERA5_DATA_DIR = Path("data/copernicus-era5-taltech")


def load_building_data(
    building_code: str,
    campus_data_dir: Optional[Path] = None,
    years: Optional[List[str]] = None,
    measurement_points: Optional[List[str]] = None,
    rename_columns: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Loads and preprocesses building energy data.
    
    Args:
        building_code: e.g. "U01"
        campus_data_dir: Base path.
        years: List of years to load.
        measurement_points: Specific points to load (optional).
        rename_columns: Whether to apply standard naming.
        verbose: Print progress.

    Returns:
        DataFrame with columns [time, heat_kwh, ..., data_status]
    """
    campus_data_dir = campus_data_dir or DEFAULT_CAMPUS_DATA_DIR
    years = years or ["2022", "2023", "2024"]

    # Load metadata to get measurement points if not provided
    # We will assume a helper to load metadata (simplified for now)
    # Ideally, we would read `andmed ulevaade.xlsx` here, but to avoid 
    # dependency on that specific file structure if arguments are passed,
    # we proceed with provided or discovered points.
    
    if measurement_points is None:
        # Check files to discover points
        # Structure: building_code/year/building_code_POINT_DETAIL_year.csv
        # e.g. U01_BHB01_näit_2022.csv
        found_points = set()
        for year in years:
            year_dir = campus_data_dir / building_code / year
            if year_dir.exists():
                for f in year_dir.glob("*.csv"):
                    # Filename: Code_Point_Detail_Year.csv
                    parts = f.stem.split("_")
                    if len(parts) >= 4:
                        # Assumption: parts[1] is the point (e.g. BHB01)
                        # We need both point and detail (e.g. näit)
                        # But traditionally we iterate points.
                        point = parts[1]
                        found_points.add(point)
        measurement_points = sorted(list(found_points))
        if verbose:
            print(f"Discovered measurement points for {building_code}: {measurement_points}")

    # Standard Details we care about
    # Usually "näit" (reading) is what we assume if we want to calc delta.
    # Or "tarbimine" (consumption). 
    # The utils loader assumed "näit" and calculated diff.
    DETAILS = ["näit", "tarbimine"] 
    
    column_data = {}
    
    for year in years:
        for point in measurement_points:
            # Try loading 'näit' first, loop through known details
            for detail in DETAILS:
                filename = f"{building_code}_{point}_{detail}_{year}.csv"
                file_path = campus_data_dir / building_code / year / filename
                
                if not file_path.exists():
                    continue
                
                # Load CSV
                # content: Time, Value
                try:
                    df_temp = pd.read_csv(file_path)
                    # Create column name: e.g. BHB01_näit
                    col_name = f"{point}_{detail}"
                    
                    # Ensure Time is datetime
                    if "Time" in df_temp.columns and "Value" in df_temp.columns:
                        df_temp["Time"] = pd.to_datetime(df_temp["Time"])
                        df_temp.set_index("Time", inplace=True)
                        
                        # Store in dictionary
                        if col_name not in column_data:
                            column_data[col_name] = []
                        column_data[col_name].append(df_temp["Value"])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    # Concatenate years for each column
    final_series_list = []
    for col_name, series_list in column_data.items():
        if not series_list:
            continue
        # Concat series
        combined = pd.concat(series_list).sort_index()
        # Handle duplicates if any (mean)
        combined = combined.groupby(combined.index).mean()
        combined.name = col_name
        final_series_list.append(combined)

    if not final_series_list:
        return pd.DataFrame() # Empty

    # Merge into one DataFrame
    df = pd.concat(final_series_list, axis=1)
    df.index.name = COL_TIME
    
    # Resample to Hourly (as per requirement)
    # Using mean for readings/values? 
    # If it is 'näit' (cumulative), we usually take last or mean?
    # Utils took mean. Let's stick to valid resampling.
    # For cumulative readings, we want the value at the hour. 
    # But usually these are hourly averages/snapshots.
    df = df.resample("1h").mean()
    
    # Calculate Deltas for "näit" columns
    # New column: BHB01_difference_kwh (or just raw delta)
    cols_to_drop = []
    for col in df.columns:
        if "näit" in col:
            # delta = diff
            delta_col = col.replace("näit", "measurement_delta_kwh") # Interim name
            # Calculate diff
            df[delta_col] = df[col].diff()
            
            # If we want to rename heavily, we can map it directly to 
            # COL_HEAT if we figure out which one is the main meter.
            # But here we just create the deltas.
            cols_to_drop.append(col)
            
    # For "tarbimine", it is already delta/consumption.
    
    # drop raw readings if wanted
    # df.drop(columns=cols_to_drop, inplace=True) 
    
    # Rename Columns
    if rename_columns:
        # Heuristic to find the MAIN heat meter
        # Usually BHB01 or BVK01?
        # Strategy: Rename columns that look like heat meters.
        # {point}_measurement_delta_kwh -> heat_kwh (if it's the main one)
        # For now, we standardize names to: {Point}_{Type}
        pass

    # Reset Index to make Time a column
    df = df.reset_index()
    
    # Initialize Data Status
    # By default measure = 0
    # Where NaN, we will later set to MISSING or Interpolate.
    df[COL_STATUS] = DataStatus.MEASURED

    return df


def load_weather_data(
    source: str = "keskkonnaportaal",
    path: Optional[Path] = None,
    era5_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Dispatcher for weather data loading.
    
    Args:
        source: "keskkonnaportaal" or "era5"
        path: Path to CSV (for keskkonnaportaal)
        era5_dir: Directory for ERA5 NetCDFs
    """
    if source == "keskkonnaportaal":
        return load_keskkonnaportaal_weather(path or DEFAULT_WEATHER_DATA_PATH_KESK)
    elif source == "era5":
        return load_copernicus_weather(era5_dir or DEFAULT_ERA5_DATA_DIR)
    else:
        raise ValueError(f"Unknown weather source: {source}")


def load_keskkonnaportaal_weather(path: Path) -> pd.DataFrame:
    """
    Loads CSV from Keskkonnaportaal.
    Standardizes column names using config.WEATHER_COL_MAPPING.
    """
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")

    # Load with semicolon delimiter
    df = pd.read_csv(path, delimiter=";")

    # Parse Time: "aasta", "kuu", "paev", "tund"
    # Note: "tund" might be 0-23 or 1-24? 
    # In utils loader: YYYY-MM-DD HH:00:00
    # Let's trust the util logic:
    # df["Time"] = pd.to_datetime(...)
    
    # Construct datetime string
    # Zfill ensures 2 digits
    time_str = (
        df["aasta - Year of measurement, UTC time"].astype(str) + "-" +
        df["kuu - Month of measurement, UTC time"].astype(str).str.zfill(2) + "-" +
        df["paev - Day of measurement, UTC time"].astype(str).str.zfill(2) + " " +
        df["tund - Hour of measurement, UTC time"].astype(str).str.zfill(2) + ":00:00"
    )
    df[COL_TIME] = pd.to_datetime(time_str)
    
    # Parse Value: Replace comma with dot
    df["value"] = (
        df["vaartus - The measured value"]
        .astype(str)
        .str.replace(",", ".")
        .astype(float)
    )
    
    # Feature codes generally local, but we have English names in another col
    # "element_nimi_eng - Element name (eng)"
    # We can pivot on English Name directly if present
    feature_col = "element_nimi_eng - Element name (eng)"
    if feature_col not in df.columns:
        feature_col = "element_kood - Element code, local"

    # Pivot
    weather_df = df.pivot_table(
        index=COL_TIME,
        columns=feature_col,
        values="value",
        aggfunc="mean"
    ).reset_index()
    
    # Rename Columns
    # Map known English names to standard names
    new_cols = {}
    for col in weather_df.columns:
        if col in WEATHER_COL_MAPPING:
            new_cols[col] = WEATHER_COL_MAPPING[col]
            
    weather_df.rename(columns=new_cols, inplace=True)
    
    # Set Index
    weather_df.set_index(COL_TIME, inplace=True)
    
    return weather_df


def load_copernicus_weather(data_dir: Path) -> pd.DataFrame:
    """
    Loads ERA5-Land NetCDF files (Surface + Clouds).
    
    CRITICAL: Handles 'ssrd' (solar radiation) accumulation reset logic.
    ERA5 accumulation resets at 01:00 UTC (step 1).
    We must diff the cumulative values but handle the reset jump.
    
    Loads ALL columns found in the datasets.
    """
    from functools import reduce

    # 1. Load Surface Files (taltech_era5land_surface_*.nc)
    surface_files = sorted(list(data_dir.glob("taltech_era5land_surface_*.nc")))
    surface_parts = []
    
    def load_netcdf_to_df(files):
        parts = []
        for f in files:
            try:
                # Load with xarray
                ds = xr.open_dataset(f)
                
                # 1. Spatial Selection
                ds = ds.isel(latitude=0, longitude=0)
                
                # 2. Expver Handling
                if "expver" in ds.dims:
                    try:
                        expvers = ds["expver"].values
                        if 1 in expvers and 5 in expvers:
                            ds_1 = ds.sel(expver=1)
                            ds_5 = ds.sel(expver=5)
                            ds = ds_1.combine_first(ds_5)
                        elif 1 in expvers:
                            ds = ds.sel(expver=1)
                        elif 5 in expvers:
                            ds = ds.sel(expver=5)
                        else:
                            ds = ds.isel(expver=0)
                        
                        # Verify expver is gone or drop it
                        if "expver" in ds.dims:
                            ds = ds.drop_dims("expver")
                        if "expver" in ds.coords:
                            ds = ds.drop_vars("expver")
                            
                    except Exception as e:
                        print(f"Expver merge failed for {f}: {e}")
                        ds = ds.isel(expver=0)

                # 3. Handle 'number' dimension
                if "number" in ds.dims:
                    ds = ds.isel(number=0)

                # 4. Valid Time Calculation (in Xarray to preserve logic, but we won't swap dims)
                if "valid_time" not in ds.variables:
                    if "step" in ds.variables and "time" in ds.variables:
                        # Calculate valid_time = time + step
                        ds = ds.assign_coords(valid_time=ds.time + ds.step)
                
                # 5. Squeeze
                ds = ds.squeeze(drop=True)

                # 6. Convert to DataFrame
                # This will likely produce either a valid_time index (if 1D) or MultiIndex (time, step)
                df_part = ds.to_dataframe().reset_index()
                
                # 7. Index Handling in Pandas
                # We want 'valid_time' to be the index 'time'
                target_col = "valid_time"
                
                if target_col not in df_part.columns:
                    # Maybe it is 'time'? If valid_time calculation failed or wasn't needed
                    if "time" in df_part.columns:
                        # Check if this 'time' is actually valid time or forecast time
                        # If step exists and is > 0, 'time' is forecast.
                        # But we calculated valid_time above.
                        pass
                
                if target_col in df_part.columns:
                    # Rename valid_time -> time
                    # But first drop conflicting 'time' column if it exists (the forecast time)
                    if COL_TIME in df_part.columns and target_col != COL_TIME:
                        df_part.drop(columns=[COL_TIME], inplace=True)
                    
                    df_part.rename(columns={target_col: COL_TIME}, inplace=True)
                    df_part.set_index(COL_TIME, inplace=True)
                    df_part.sort_index(inplace=True)
                else:
                    # Fallback: Use 'time' if it exists
                    if "time" in df_part.columns:
                        df_part.rename(columns={"time": COL_TIME}, inplace=True)
                        df_part.set_index(COL_TIME, inplace=True)
                        df_part.sort_index(inplace=True)

                # 8. Cleanup Columns
                cols_to_drop = [c for c in ["number", "expver", "latitude", "longitude", "step", "surface"] if c in df_part.columns]
                df_part.drop(columns=cols_to_drop, inplace=True, errors="ignore")
                
                # Remove duplicates
                df_part = df_part[~df_part.index.duplicated(keep='first')]

                parts.append(df_part)

            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts).sort_index()

    df_surface = load_netcdf_to_df(surface_files)
    
    # 2. Load Cloud Files (taltech_era5_clouds_*.nc)
    cloud_files = sorted(list(data_dir.glob("taltech_era5_clouds_*.nc")))
    df_clouds = load_netcdf_to_df(cloud_files)
    
    # 3. Merge them
    # Use outer join to keep all times? Or inner? Assuming valid matching times.
    # Check if empty
    dfs_to_merge = []
    if not df_surface.empty:
        dfs_to_merge.append(df_surface)
    if not df_clouds.empty:
        dfs_to_merge.append(df_clouds)
        
    if not dfs_to_merge:
        raise FileNotFoundError(f"No valid ERA5 .nc files found in {data_dir}")
        
    if len(dfs_to_merge) > 1:
        # Merge on index (time)
        full_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs_to_merge)
    else:
        full_df = dfs_to_merge[0]

    # ... Solar Radiation Logic ...
    if "ssrd" in full_df.columns:
        ssrd_raw = full_df["ssrd"]
        # Handle resets same as before
        ssrd_diff = ssrd_raw.diff()
        is_reset = ssrd_diff < -3600
        ssrd_hourly = ssrd_diff.copy()
        ssrd_hourly[is_reset] = ssrd_raw[is_reset]
        ssrd_hourly = ssrd_hourly.clip(lower=0)
        
        # Add converted column, keep original too if requested (user asked to load everything)
        # But we must rename standard ones to ensure pipeline works.
        full_df[COL_SOLAR] = ssrd_hourly / 3600.0
    
    # Conversions
    if "t2m" in full_df.columns:
        full_df[COL_TEMP_OUT] = full_df["t2m"] - 273.15
        
    if "u10" in full_df.columns and "v10" in full_df.columns:
        full_df[COL_WIND_SPEED] = np.sqrt(full_df["u10"]**2 + full_df["v10"]**2)
    
    # Do NOT filter columns arbitrarily. Return everything.
    return full_df
