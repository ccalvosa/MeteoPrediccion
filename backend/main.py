"""
MeteoEnsemble API v3
Usa nubosidad y tipo de precipitación para símbolo más preciso.
"""

import json
import math
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from data import get_meteogram_data

app = FastAPI(title="MeteoEnsemble API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─── Serialización nan-safe ───────────────────────────────────────────────────
def _safe(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if not math.isfinite(f) else round(f, 3)


def _json(obj) -> Response:
    body = json.dumps(obj, allow_nan=False, default=lambda x: None)
    return Response(content=body, media_type="application/json")


# ─── Estadísticas del ensemble ────────────────────────────────────────────────
def _ens_stats(arr: np.ndarray) -> dict | None:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return None
    return {
        "median": _safe(np.median(valid)),
        "p025":   _safe(np.percentile(valid, 2.5)),
        "p975":   _safe(np.percentile(valid, 97.5)),
        "p10":    _safe(np.percentile(valid, 10)),
        "p90":    _safe(np.percentile(valid, 90)),
        "mean":   _safe(np.mean(valid)),
        "std":    _safe(np.std(valid)),
        "n":      int(valid.size),
    }


def _prob_above(arr: np.ndarray, thr: float) -> float:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0
    return round(float(np.sum(valid > thr) / valid.size * 100), 1)


def _prob_flag(arr2d: np.ndarray) -> float:
    """% miembros con flag=1 en al menos un paso del día. arr2d: (n_steps, n_members)."""
    valid = np.nan_to_num(arr2d, nan=0.0)
    any_hit = (valid > 0.5).any(axis=0)  # (n_members,)
    return round(float(any_hit.mean() * 100), 1)


# ─── Símbolo meteorológico ────────────────────────────────────────────────────
def _symbol(precip_prob, cloud_mean, thunder_prob, snow_prob,
            rain_prob, drizzle_prob, precip_p90) -> str:
    if precip_prob > 60:
        if thunder_prob > 25 or precip_p90 > 15:
            return "thunderstorm"
        if snow_prob > 50:
            return "snow"
        if snow_prob > 20:
            return "sleet"
        if precip_prob > 80:
            return "heavy_rain"
        return "rain"
    if precip_prob > 20 or drizzle_prob > 15:
        return "drizzle"
    if cloud_mean > 80:
        return "cloudy"
    if cloud_mean > 40:
        return "partly_cloudy"
    return "sunny"


# ─── Reconstruir array (n_lead, n_member) ────────────────────────────────────
def _to_array(members_dict: dict) -> np.ndarray:
    cols = []
    for m in sorted(members_dict.keys(), key=int):
        vals = members_dict[m]
        cols.append([np.nan if v is None else v for v in vals])
    return np.array(cols).T  # (n_lead, n_member)


# ─── Agregación diaria ────────────────────────────────────────────────────────
def _aggregate(raw: dict) -> list:
    vt  = pd.to_datetime(raw["temperature_2m"]["valid_time"], utc=True)
    dates = sorted(set(vt.date))

    t_arr  = _to_array(raw["temperature_2m"]["members"])
    pcp_arr= _to_array(raw["precipitation"]["members"])
    w_arr  = _to_array(raw["wind_10m"]["members"])
    cc_arr = _to_array(raw["cloud_cover"]["members"])

    pt = raw.get("precip_type", {})
    rain_arr  = _to_array(pt["rain"]["members"])        if pt.get("rain")         else None
    snow_arr  = _to_array(pt["snow"]["members"])        if pt.get("snow")         else None
    thun_arr  = _to_array(pt["thunderstorm"]["members"])if pt.get("thunderstorm") else None
    driz_arr  = _to_array(pt["drizzle"]["members"])     if pt.get("drizzle")      else None

    daily = []
    for date in dates:
        idx = [i for i, d in enumerate(vt) if d.date() == date]
        if not idx:
            continue

        # Temperatura
        t_day      = t_arr[idx, :]
        tmax_stats = _ens_stats(np.nanmax(t_day, axis=0))
        tmin_stats = _ens_stats(np.nanmin(t_day, axis=0))

        # Precipitación acumulada — paso real calculado desde valid_times
        steps_h = [3.0] + [
            float((vt[idx[i]] - vt[idx[i-1]]).total_seconds() / 3600)
            for i in range(1, len(idx))
        ]
        sh        = np.array(steps_h)[:, np.newaxis]
        pcp_accum = np.nansum(pcp_arr[idx, :] * sh, axis=0)
        pcp_stats  = _ens_stats(pcp_accum)
        precip_prob = _prob_above(pcp_accum, 0.5)
        pcp_p90     = float(pcp_stats["p90"]) if pcp_stats and pcp_stats["p90"] else 0.0

        # Viento máximo diario
        wind_stats = _ens_stats(np.nanmax(w_arr[idx, :], axis=0))

        # Nubosidad media diaria
        cc_day   = cc_arr[idx, :]
        cc_mean  = float(np.nanmean(cc_day))
        cc_stats = _ens_stats(np.nanmean(cc_day, axis=0))

        # Probabilidades de tipo
        def _pp(arr): return _prob_flag(arr[idx, :]) if arr is not None else 0.0
        rain_prob    = _pp(rain_arr)
        snow_prob    = _pp(snow_arr)
        thunder_prob = _pp(thun_arr)
        drizzle_prob = _pp(driz_arr)

        symbol = _symbol(precip_prob, cc_mean, thunder_prob, snow_prob,
                         rain_prob, drizzle_prob, pcp_p90)

        daily.append({
            "date":         date.isoformat(),
            "symbol":       symbol,
            "temp_max":     tmax_stats,
            "temp_min":     tmin_stats,
            "precip_mm":    pcp_stats,
            "precip_prob":  precip_prob,
            "wind_max":     wind_stats,
            "wind_label":   raw["wind_10m"].get("label", "Viento 10m"),
            "cloud_cover":  cc_stats,
            "rain_prob":    rain_prob,
            "snow_prob":    snow_prob,
            "thunder_prob": thunder_prob,
        })

    return daily


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "models": ["ecmwf", "gefs"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/forecast")
def forecast(
    lat:   float = Query(..., ge=-90,  le=90),
    lon:   float = Query(..., ge=-180, le=180),
    model: str   = Query("ecmwf", pattern="^(ecmwf|gefs)$"),
):
    try:
        raw   = get_meteogram_data(lat, lon, model)
        daily = _aggregate(raw)
        meta  = raw["metadata"]
        payload = {
            "status":      "ok",
            "model_label": meta["model"],
            "init_time":   meta["init_time"],
            "n_members":   meta["n_members"],
            "nearest_grid_point": {
                "lat": meta["nearest_lat"],
                "lon": meta["nearest_lon"],
            },
            "source":   meta["source"],
            "forecast": daily,
        }
        return _json(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
