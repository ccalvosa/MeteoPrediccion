import xarray as xr
import numpy as np
import pandas as pd
from functools import lru_cache

# ─── URLs ────────────────────────────────────────────────────────────────────
ZARR_ECMWF = "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr"
ZARR_GEFS  = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"

MAX_LEAD_HOURS = 360

VARS_ECMWF = [
    "temperature_2m",
    "precipitation_surface",
    "wind_gust_10m",
    "total_cloud_cover_atmosphere",
    "categorical_precipitation_type_surface",   # 0=nada,1=lluvia,2=tormenta,5=nieve,6=nieve húmeda
]
VARS_GEFS = [
    "temperature_2m",
    "precipitation_surface",
    "wind_u_10m",
    "wind_v_10m",
    "total_cloud_cover_atmosphere",
    "categorical_rain_surface",   # binario 0/1
    "categorical_snow_surface",   # binario 0/1
]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _clean(arr) -> list:
    return [None if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))))
            else float(x) for x in arr]


def _members_dict(data2d, member_coords) -> dict:
    """Array (n_lead, n_member) → dict {member_id: [valores]}."""
    return {
        str(int(m)): _clean(data2d[:, i])
        for i, m in enumerate(member_coords)
    }


def _valid_times(latest_init: pd.Timestamp, lead_times) -> list:
    return [(latest_init + pd.Timedelta(lt)).isoformat() for lt in lead_times]


def _precip_mmh(prate: np.ndarray) -> np.ndarray:
    return np.maximum(prate * 3600, 0)


# ─── Apertura de datasets (cacheados) ────────────────────────────────────────
@lru_cache(maxsize=1)
def _open_ecmwf() -> xr.Dataset:
    print("Abriendo dataset ECMWF IFS ENS...")
    return xr.open_zarr(ZARR_ECMWF, consolidated=True)


@lru_cache(maxsize=1)
def _open_gefs() -> xr.Dataset:
    print("Abriendo dataset NOAA GEFS...")
    return xr.open_zarr(ZARR_GEFS, consolidated=True)


# ─── Función principal ───────────────────────────────────────────────────────
def get_meteogram_data(lat: float, lon: float, model: str) -> dict:
    if model == "ecmwf":
        return _get_ecmwf(lat, lon)
    elif model == "gefs":
        return _get_gefs(lat, lon)
    else:
        raise ValueError(f"Modelo no soportado: {model}. Usa 'ecmwf' o 'gefs'.")


# ─── ECMWF IFS ENS ───────────────────────────────────────────────────────────
def _get_ecmwf(lat: float, lon: float) -> dict:
    ds = _open_ecmwf()
    latest_init = pd.Timestamp(ds.init_time.values[-1])

    point = ds[VARS_ECMWF].sel(
        init_time=latest_init,
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).compute()

    real_lat = float(point.latitude)
    real_lon = float(point.longitude)
    lead_times = point.lead_time.values

    mask = np.array([pd.Timedelta(lt).total_seconds() / 3600 <= MAX_LEAD_HOURS
                     for lt in lead_times])
    lead_times  = lead_times[mask]
    valid_times = _valid_times(latest_init, lead_times)
    members     = point.ensemble_member.values
    n_members   = len(members)

    # Temperatura
    t2m = point["temperature_2m"].values[mask]
    temperature = {"valid_time": valid_times, "members": _members_dict(t2m, members), "units": "°C"}

    # Precipitación
    pcp = _precip_mmh(point["precipitation_surface"].values[mask])
    precipitation = {"valid_time": valid_times, "members": _members_dict(pcp, members), "units": "mm/h"}

    # Racha de viento
    gust = point["wind_gust_10m"].values[mask] * 3.6
    wind = {
        "valid_time": valid_times,
        "members": _members_dict(gust, members),
        "units": "km/h",
        "label": "Racha máx. 10m",
    }

    # Nubosidad (0–100 %)
    cloud = point["total_cloud_cover_atmosphere"].values[mask]
    cloud_cover = {"valid_time": valid_times, "members": _members_dict(cloud, members), "units": "%"}

    # Tipo de precipitación → flags binarios por tipo
    # IFS ptype: 0=nada, 1=lluvia, 2=tormenta, 3=lluvia engelante,
    #            4=mixto/hielo, 5=nieve, 6=nieve húmeda, 9=granizo, 11=llovizna
    ptype = point["categorical_precipitation_type_surface"].values[mask]  # (lead, member)
    precip_type = {
        "valid_time": valid_times,
        "rain":        {"members": _members_dict((ptype == 1).astype(float), members)},
        "thunderstorm":{"members": _members_dict((ptype == 2).astype(float), members)},
        "snow":        {"members": _members_dict(((ptype == 5) | (ptype == 6)).astype(float), members)},
        "drizzle":     {"members": _members_dict((ptype == 11).astype(float), members)},
        "freezing":    {"members": _members_dict((ptype == 3).astype(float), members)},
    }

    return {
        "metadata": {
            "model": "ECMWF IFS ENS",
            "init_time": latest_init.isoformat(),
            "requested_lat": lat, "requested_lon": lon,
            "nearest_lat": real_lat, "nearest_lon": real_lon,
            "n_members": n_members,
            "n_lead_times": len(valid_times),
            "source": "ECMWF IFS ENS via dynamical.org",
        },
        "temperature_2m": temperature,
        "precipitation":  precipitation,
        "wind_10m":       wind,
        "cloud_cover":    cloud_cover,
        "precip_type":    precip_type,
    }


# ─── NOAA GEFS ───────────────────────────────────────────────────────────────
def _get_gefs(lat: float, lon: float) -> dict:
    ds = _open_gefs()
    latest_init = pd.Timestamp(ds.init_time.values[-1])

    max_lead  = pd.Timedelta(hours=MAX_LEAD_HOURS)
    lead_mask = ds.lead_time <= max_lead

    point = ds[VARS_GEFS].sel(
        init_time=latest_init,
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).isel(lead_time=lead_mask).compute()

    point = point.transpose("lead_time", "ensemble_member")

    real_lat    = float(point.latitude)
    real_lon    = float(point.longitude)
    lead_times  = point.lead_time.values
    valid_times = _valid_times(latest_init, lead_times)
    members     = point.ensemble_member.values
    n_members   = len(members)

    # Temperatura
    t2m = point["temperature_2m"].values
    temperature = {"valid_time": valid_times, "members": _members_dict(t2m, members), "units": "°C"}

    # Precipitación
    pcp = _precip_mmh(point["precipitation_surface"].values)
    precipitation = {"valid_time": valid_times, "members": _members_dict(pcp, members), "units": "mm/h"}

    # Viento módulo
    u    = point["wind_u_10m"].values
    v    = point["wind_v_10m"].values
    wspd = np.sqrt(u**2 + v**2) * 3.6
    wind = {
        "valid_time": valid_times,
        "members": _members_dict(wspd, members),
        "units": "km/h",
        "label": "Viento medio 10m",
    }

    # Nubosidad
    cloud = point["total_cloud_cover_atmosphere"].values
    cloud_cover = {"valid_time": valid_times, "members": _members_dict(cloud, members), "units": "%"}

    # Tipo de precipitación — GEFS tiene flags binarios directos
    rain_flag = point["categorical_rain_surface"].values
    snow_flag = point["categorical_snow_surface"].values
    precip_type = {
        "valid_time": valid_times,
        "rain":        {"members": _members_dict(rain_flag.astype(float), members)},
        "thunderstorm":{"members": _members_dict(np.zeros_like(rain_flag, dtype=float), members)},  # GEFS no tiene flag de tormenta directo
        "snow":        {"members": _members_dict(snow_flag.astype(float), members)},
        "drizzle":     {"members": _members_dict(np.zeros_like(rain_flag, dtype=float), members)},
        "freezing":    {"members": _members_dict(np.zeros_like(rain_flag, dtype=float), members)},
    }

    return {
        "metadata": {
            "model": "NOAA GEFS",
            "init_time": latest_init.isoformat(),
            "requested_lat": lat, "requested_lon": lon,
            "nearest_lat": real_lat, "nearest_lon": real_lon,
            "n_members": n_members,
            "n_lead_times": len(valid_times),
            "source": "NOAA GEFS via dynamical.org",
        },
        "temperature_2m": temperature,
        "precipitation":  precipitation,
        "wind_10m":       wind,
        "cloud_cover":    cloud_cover,
        "precip_type":    precip_type,
    }
