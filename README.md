# MeteoEnsemble v2 🌩️⚡

Web-app de predicción probabilística con soporte multi-modelo:
**GEFS** (31 mbrs, 10 días) · **IFS ENS** (51 mbrs, 15 días) · **Combinado** (82 mbrs, 15 días)

Fuente: [dynamical.org](https://dynamical.org) · CC BY 4.0

---

## Diferencias entre modelos

| | GEFS | IFS ENS | Combinado |
|---|---|---|---|
| Centro | NOAA/NCEP | ECMWF | — |
| Miembros | 31 | 51 | 82 |
| Horizonte | 35 días (aquí 10) | 15 días | 15 días |
| Resolución | 0.25° | 0.25° | 0.25° |
| Paso | 3h (0–240h) | 3h (0–144h), 6h (144–360h) | heredado |
| Disponible desde | 2020-10-01 | 2024-04-01 | 2024-04-01 |
| Tipo precip. | binario rain/snow | código numérico (0–14) | heredado |
| Racha viento | ✗ | ✓ (`wind_gust_10m`) | ✓ |
| Tormenta flag | ✗ (proxy precip.) | ✓ (`ptype==2`) | ✓ |

## Modo combinado (super-ensemble)

Los 82 miembros (31 GEFS + 51 IFS) se tratan como un único ensemble.
- Se alinean temporalmente por `valid_time` (tolerancia ±3.5h)
- Solo se combinan las variables comunes a ambos modelos
- El IC95% refleja la incertidumbre total incluyendo la diversidad inter-modelo
- El símbolo se determina sobre el pool completo de 82 miembros

---

## Arquitectura

```
GitHub Pages (frontend/index.html)
        │  GET /forecast?lat=&lon=&days=&model=[gefs|ifs|combined]
        ▼
Render (backend/main.py · FastAPI)
        │        │
        │        ├── xr.open_zarr(GEFS_URL)   [31 mbrs, 35d]
        │        └── xr.open_zarr(IFS_URL)    [51 mbrs, 15d]
        ▼
dynamical.org (Zarr en Source Cooperative)
```

---

## Despliegue

### Backend en Render

1. Fork del repo en GitHub
2. Crear Web Service en render.com → detecta `render.yaml` automáticamente
3. Anotar la URL pública del servicio

### Frontend en GitHub Pages

1. Editar `frontend/index.html` línea ~15: cambiar `API_BASE`
2. GitHub → Settings → Pages → `/frontend`

> ⚠️ **RAM**: El modo combinado abre dos datasets Zarr simultáneamente.
> En plan Free de Render (512 MB) puede haber OOM en picos. Plan Starter recomendado.

---

## Variables por modelo

### GEFS
- `temperature_2m`, `maximum_temperature_2m`, `minimum_temperature_2m`
- `precipitation_surface` → mm/3h
- `wind_u_10m`, `wind_v_10m` → velocidad + dirección
- `total_cloud_cover_atmosphere`, `relative_humidity_2m`
- `pressure_reduced_to_mean_sea_level`
- `categorical_rain_surface`, `categorical_snow_surface` (binarios)

### IFS ENS (adicionales)
- `wind_gust_10m` → racha de viento
- `categorical_precipitation_type_surface` → código 0-14 (1=lluvia, 2=tormenta, 5=nieve, 6=nieve húmeda)
- `dew_point_temperature_2m` → se convierte a HR vía Magnus
- `geopotential_height_500hpa`, `geopotential_height_850hpa`

---

## Lógica de símbolos

```python
if P(precip) > 60%:
    if P(tormenta) > 30% or proxy_tormenta > 40%:  → ⛈️
    elif P(nieve) > 50%:                             → ❄️
    elif P(nieve) > 20%:                             → 🌨️
    elif P(precip) > 80%:                            → 🌧️ (intensa)
    else:                                            → 🌧️
elif P(precip) > 20%:                                → 🌦️
elif nubosidad_media > 75%:                          → ☁️
elif nubosidad_media > 40%:                          → ⛅
else:                                                → ☀️
```
