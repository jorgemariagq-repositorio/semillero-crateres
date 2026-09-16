"""
craterscore_complejo.py — Modelo de perfil para crateres COMPLEJOS.

Extiende `craterscore.py`. No lo reemplaza: se importa aparte y el modulo
original sigue funcionando igual.

POR QUE EXISTE
El modelo de `craterscore.crater_profile` describe un crater SIMPLE: cuenco
hundido en el centro y borde elevado. La mayoria de las estructuras de impacto
terrestres por encima de unos 4 km no son asi: son COMPLEJAS, con un
levantamiento central que la erosion deja en resalte, un foso anular alrededor
y un borde atenuado.

Serra da Cangalha lo demostro en la primera corrida con dato real. Su perfil
radial medio sobre GLO-30 tiene +195 m en el centro, un minimo de -25 m hacia
los 3,5 km y un realce de unos +12 m a 6,5 km. El modelo simple daba R2 = 0,19
y un contraste anular NEGATIVO, que no era un error de medicion sino la firma
correcta de una morfologia que el modelo no contemplaba.

QUE CAMBIA
  1. Modelo con termino de levantamiento central (Ec. del perfil complejo).
  2. Posibilidad de FIJAR r_rim al valor de catalogo. Dejandolo libre, el
     ajuste se engancha al anillo interno (Serra da Cangalha tiene uno a
     1,6 km de radio) y reporta un diametro cinco veces menor.
  3. Contraste foso-borde, que es positivo tanto para crateres simples como
     complejos, a diferencia del contraste centro-borde.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from craterscore import (basin_circularity, crater_profile,
                         azimuthal_coherence, detrend_plane, radial_profiles,
                         sublevel_basin_mask)

__all__ = ["perfil_complejo", "ajustar_perfil_complejo", "contraste_foso",
           "metricas_complejas", "crater_complejo_sintetico"]

_EPS = 1e-9


def perfil_complejo(r, z0, dh_peak, sigma_p, dh_moat, r_moat, sigma_m,
                    dh_rim, r_rim, sigma_r):
    """Perfil radial de un crater complejo.

        z(r) = z0
             + dh_peak * exp(-r^2 / 2 sigma_p^2)              levantamiento central
             - dh_moat * exp(-(r - r_moat)^2 / 2 sigma_m^2)   foso anular
             + dh_rim  * exp(-(r - r_rim)^2  / 2 sigma_r^2)   borde

    Con dh_peak = 0 se reduce a un crater simple de piso plano. Todas las
    amplitudes son >= 0: los signos estan en la formula, no en los parametros.
    """
    r = np.asarray(r, dtype=float)
    return (z0
            + dh_peak * np.exp(-(r ** 2) / (2.0 * sigma_p ** 2 + _EPS))
            - dh_moat * np.exp(-((r - r_moat) ** 2) / (2.0 * sigma_m ** 2 + _EPS))
            + dh_rim * np.exp(-((r - r_rim) ** 2) / (2.0 * sigma_r ** 2 + _EPS)))


def crater_complejo_sintetico(shape=(512, 512), r_rim=100.0, dh_peak=150.0,
                              sigma_p=18.0, dh_moat=25.0, r_moat=55.0,
                              sigma_m=25.0, dh_rim=12.0, sigma_r=20.0,
                              z0=0.0, noise_std=0.0, seed=None):
    """DEM sintetico de un crater complejo, para pruebas."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    rr = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    dem = perfil_complejo(rr, z0, dh_peak, sigma_p, dh_moat, r_moat, sigma_m,
                          dh_rim, r_rim, sigma_r)
    if noise_std > 0:
        dem = dem + np.random.default_rng(seed).normal(0, noise_std, dem.shape)
    return dem


def ajustar_perfil_complejo(r, z, r_rim_conocido=None, tolerancia=0.25):
    """Ajusta `perfil_complejo` al perfil radial medio.

    Si se pasa `r_rim_conocido`, el radio del borde queda acotado a
    +-`tolerancia` alrededor de ese valor. Dejarlo libre es lo que hace que el
    ajuste se enganche a un anillo interno y reporte un diametro equivocado.

    Devuelve (popt, r2). Si no converge, (None, 0.0).
    """
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    bien = np.isfinite(z)
    if bien.sum() < 20:
        return None, 0.0
    r, z = r[bien], z[bien]

    relieve = float(np.nanmax(z) - np.nanmin(z))
    if relieve < _EPS:
        return None, 0.0
    r_max = float(r.max())

    if r_rim_conocido:
        lo_rim = max(r_rim_conocido * (1 - tolerancia), 0.05 * r_max)
        hi_rim = min(r_rim_conocido * (1 + tolerancia), 0.98 * r_max)
        p_rim = float(np.clip(r_rim_conocido, lo_rim + _EPS, hi_rim - _EPS))
    else:
        lo_rim, hi_rim = 0.10 * r_max, 0.95 * r_max
        p_rim = 0.5 * r_max

    p0 = [float(np.median(z)),
          max(float(z[r < 0.1 * r_max].mean() - np.median(z)), 1.0),
          max(0.15 * p_rim, 1.0),
          max(relieve * 0.1, 1.0), 0.5 * p_rim, max(0.25 * p_rim, 1.0),
          max(relieve * 0.1, 1.0), p_rim, max(0.15 * p_rim, 1.0)]

    lo = [float(z.min()) - 5 * relieve, 0.0, 0.5,
          0.0, 0.05 * r_max, 0.5,
          0.0, lo_rim, 0.5]
    hi = [float(z.max()) + 5 * relieve, 10 * relieve, r_max,
          10 * relieve, 0.95 * r_max, r_max,
          10 * relieve, hi_rim, r_max]
    p0 = [float(np.clip(v, l + _EPS, h - _EPS)) for v, l, h in zip(p0, lo, hi)]

    try:
        popt, _ = curve_fit(perfil_complejo, r, z, p0=p0, bounds=(lo, hi),
                            maxfev=40000)
    except (RuntimeError, ValueError):
        return None, 0.0

    resid = z - perfil_complejo(r, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0
    return popt, float(np.clip(r2, 0.0, 1.0))


def contraste_foso(r, perfil_medio, r_rim, ancho=0.2):
    """Realce del borde sobre el foso anular, en metros.

    El contraste centro-borde de `craterscore.ring_contrast` es NEGATIVO en un
    crater complejo, porque el levantamiento central esta mas alto que el
    borde. Este descriptor compara el borde con el MINIMO interior, y es
    positivo para las dos morfologias.

    Devuelve (realce_m, r_foso_km).
    """
    r = np.asarray(r, float)
    z = np.asarray(perfil_medio, float)

    dentro = (r > 0.25 * r_rim) & (r < 0.85 * r_rim) & np.isfinite(z)
    banda = (np.abs(r - r_rim) <= ancho * r_rim) & np.isfinite(z)
    if dentro.sum() < 3 or banda.sum() < 3:
        return float("nan"), float("nan")

    k = int(np.nanargmin(np.where(dentro, z, np.inf)))
    return float(np.mean(z[banda]) - z[k]), float(r[k])


def metricas_complejas(dem, cx=None, cy=None, r_max=None, n_az=36,
                       r_rim_conocido=None, detrend=True):
    """Metricas morfometricas con el modelo de crater complejo.

    `r_rim_conocido` va en las MISMAS unidades que el DEM (pixeles). Cuando se
    conoce el diametro de catalogo, pasarlo evita que el ajuste se enganche a
    un anillo interno.

    Devuelve las mismas claves que `craterscore.morphometric_metrics`, mas:
      dh_peak        amplitud del levantamiento central
      dh_moat        profundidad del foso
      r_moat_px      radio del foso segun el ajuste
      realce_borde   realce del borde sobre el foso, en unidades del DEM
      r_foso_px      radio del minimo interior observado
      tipo           "complejo" si dh_peak domina, "simple" si no
    """
    dem = np.asarray(dem, dtype=float)
    ny, nx = dem.shape
    cx = (nx - 1) / 2.0 if cx is None else cx
    cy = (ny - 1) / 2.0 if cy is None else cy
    r_max = 0.45 * min(nx, ny) if r_max is None else r_max

    work, _ = detrend_plane(dem) if detrend else (dem.copy(), None)
    r, prof = radial_profiles(work, cx, cy, r_max, n_az=n_az)
    medio = np.nanmean(prof, axis=0)

    popt, r2 = ajustar_perfil_complejo(r, medio, r_rim_conocido)
    vacio = {"r2": 0.0, "r_rim_px": np.nan, "dh_peak": np.nan,
             "dh_moat": np.nan, "dh_rim": np.nan, "r_moat_px": np.nan,
             "coherence": 0.0, "ring_contrast": 0.0, "realce_borde": np.nan,
             "r_foso_px": np.nan, "bci": 0.0, "tipo": "indefinido",
             "edge_limited": True}
    if popt is None:
        return vacio

    z0, dh_peak, sigma_p, dh_moat, r_moat, sigma_m, dh_rim, r_rim, sigma_r = popt

    realce, r_foso = contraste_foso(r, medio, r_rim)
    mask = sublevel_basin_mask(work, cx, cy, r_max=1.2 * r_rim, level=z0)

    # Contraste centro-borde clasico, que en un complejo sale negativo.
    from craterscore import ring_contrast as _ring
    kring = _ring(work, cx, cy, r_rim)

    return {
        "r2": r2,
        "r_rim_px": float(r_rim),
        "dh_peak": float(dh_peak),
        "dh_moat": float(dh_moat),
        "dh_rim": float(dh_rim),
        "r_moat_px": float(r_moat),
        "coherence": azimuthal_coherence(r, prof, r_rim),
        "ring_contrast": kring,
        "realce_borde": realce,
        "r_foso_px": r_foso,
        "bci": basin_circularity(mask, method="contour"),
        "tipo": "complejo" if dh_peak > 2.0 * max(dh_rim, _EPS) else "simple",
        "edge_limited": bool(r_rim > 0.8 * r_max),
    }
