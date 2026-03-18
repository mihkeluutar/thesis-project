import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd() / "thesis-project"
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean"
WEATHER_FILE = CLEAN_DATA_DIR / "weather_features_hourly.csv"

weather_df = pd.read_csv(WEATHER_FILE, parse_dates=["datetime"]).set_index("datetime")

def load_and_preprocess_building_data(building_code: str):
    filepath = CLEAN_DATA_DIR / f"{building_code}_hourly_merged.csv"
    df = pd.read_csv(filepath, parse_dates=["Time"]).rename(columns={"Time": "datetime"}).set_index("datetime")
    heat_cols = [c for c in df.columns if '__space_heating__' in c and 'mwh' in c]
    if not heat_cols: heat_cols = [c for c in df.columns if '__total__' in c and 'mwh' in c]
    target_col = heat_cols[0] 
    model_df = df[[target_col]].copy().rename(columns={target_col: "heat_mwh"})
    model_df["heat_kwh"] = model_df["heat_mwh"] * 1000.0
    model_df = model_df.join(weather_df, how="inner")
    
    model_df.loc[model_df["heat_kwh"] <= 0, "heat_kwh"] = np.nan
    Q1 = model_df['heat_kwh'].quantile(0.01)
    Q3 = model_df['heat_kwh'].quantile(0.99)
    model_df.loc[model_df["heat_kwh"] > (Q3 + 3 * (Q3 - Q1)), "heat_kwh"] = np.nan
    model_df = model_df.interpolate(method="time", limit_direction="both").dropna()
    return model_df

b = "U06"
building_df = load_and_preprocess_building_data(b)

building_df["heat_lag1"] = building_df["heat_kwh"].shift(1)
building_df["heat_lag2"] = building_df["heat_kwh"].shift(2)
building_df = building_df.dropna()

split_date = "2023-12-31 23:59:59"

# Train only on 2023
train_df = building_df[(building_df.index >= "2023-01-01") & (building_df.index <= split_date)].copy()
test_df = building_df[building_df.index > split_date].copy()

heating_season_train = train_df["COP_temp_c"] < 15.0
heating_season_test = test_df["COP_temp_c"] < 15.0
target = "heat_kwh"
exog_cols = ["COP_temp_c", "COP_wind_speed_ms", "COP_ssrd_W_per_m2"]

def extract_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"R2": r2, "MAE": mae, "RMSE": rmse}

# Persistence
y_pred_persist = test_df["heat_lag1"]
print("Persistence:", extract_metrics(test_df.loc[heating_season_test, target], y_pred_persist[heating_season_test]))

# Static ES
train_es = train_df[heating_season_train]
es_model = LinearRegression().fit(train_es[exog_cols], train_es[target])
print("Static ES Coefs:", dict(zip(exog_cols, es_model.coef_)))
y_pred_es = pd.Series(es_model.predict(test_df[exog_cols]), index=test_df.index)
print("Static ES:", extract_metrics(test_df.loc[heating_season_test, target], y_pred_es[heating_season_test]))

# ARX
arx_cols = exog_cols + ["heat_lag1", "heat_lag2"]
train_arx = train_df[heating_season_train]
arx_model = LinearRegression().fit(train_arx[arx_cols], train_arx[target])
print("ARX Coefs:", dict(zip(arx_cols, arx_model.coef_)))
y_pred_arx = pd.Series(arx_model.predict(test_df[arx_cols]), index=test_df.index)
print("ARX ES:", extract_metrics(test_df.loc[heating_season_test, target], y_pred_arx[heating_season_test]))

