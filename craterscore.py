"""
craterscore.py — Métricas morfométricas para estructuras circulares en DEM.

Semillero de detección de estructuras de impacto — Fase 0 (compuerta).
Este módulo NO detecta cráteres. Calcula métricas descriptivas sobre una
ventana de DEM centrada en un candidato, para que después se evalúe su
poder discriminante contra un conjunto etiquetado.

Correcciones explícitas respecto al borrador v2.0 del documento:

  [C1] Signo del modelo de perfil radial. La Ec. 2 del borrador resta la
       amplitud del borde y suma la del piso, lo que describe un centro
       elevado rodeado de un anillo deprimido (un "anticráter"). Aquí el
       borde suma y el piso resta. Ver `crater_profile`.

  [C2] BCI sobre máscara real. El código del borrador pasaba una ventana
       rectangular llena de True a la función de circularidad, con lo cual
       el índice valía pi/4 = 0.785 para todo candidato, independientemente
       del terreno. Aquí la máscara se deriva del DEM. Ver
       `sublevel_basin_mask`.

  [C3] Sesgo de perímetro rasterizado. Contar píxeles de borde sobrestima
       el perímetro de un disco en un factor cercano a 4/pi, deprimiendo el
       BCI de un círculo perfecto a ~0.6 y volviendo inalcanzable el umbral
       de 0.7 del texto. `basin_circularity` usa por defecto el perímetro
       del contorno vectorizado. El método sesgado se conserva como
       `method="pixel"` para poder documentar el sesgo.

  [C4] Muestreo de perfiles radiales por interpolación bilineal en lugar de
       truncamiento a entero, que introduce aliasing azimutal.

  [C5] Detrending planar previo. Sin él, el R^2 del ajuste está dominado por
       la pendiente regional y no por la señal de cráter.

Dependencias: numpy, scipy, scikit-image. Sin GEE, sin red.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_fill_holes, binary_opening, gaussian_filter,
                           label, map_coordinates)
from scipy.optimize import curve_fit
from skimage.measure import find_contours

__all__ = [
    "crater_profile",
    "synthetic_crater",
    "detrend_plane",
    "radial_profiles",
    "fit_radial_profile",
    "azimuthal_coherence",
    "ring_contrast",
    "sublevel_basin_mask",
    "basin_circularity",
    "morphometric_metrics",
    "BCI_DISC_CEILING",
]

_EPS = 1e-9

#: BCI de un disco rasterizado perfecto con method="contour". Techo empírico.
BCI_DISC_CEILING = 0.894


# ----------------------------------------------------------------------
# Modelo de perfil
# ----------------------------------------------------------------------
def crater_profile(r, z0, dh_rim, r_rim, sigma_r, dh_floor, sigma_f):
    """Perfil radial analítico de un cráter.

    [C1] Borde ELEVADO en r = r_rim (término positivo), piso DEPRIMIDO en
    r = 0 (término negativo). Con dh_rim, dh_floor >= 0 el modelo describe
    un cráter; el borrador tenía los signos invertidos.

    Parámetros
    ----------
    r        : distancia radial al centro [px o m, consistente con sigma]
    z0       : elevación de referencia del entorno
    dh_rim   : altura del borde sobre el entorno (>= 0)
    r_rim    : radio del borde
    sigma_r  : anchura gaussiana del borde
    dh_floor : profundidad del piso bajo el entorno (>= 0)
    sigma_f  : anchura gaussiana del piso
    """
    r = np.asarray(r, dtype=float)
    rim = dh_rim * np.exp(-((r - r_rim) ** 2) / (2.0 * sigma_r ** 2 + _EPS))
    floor = dh_floor * np.exp(-(r ** 2) / (2.0 * sigma_f ** 2 + _EPS))
    return z0 + rim - floor


def synthetic_crater(
    shape=(512, 512),
    center=None,
    r_rim=60.0,
    dh_rim=40.0,
    dh_floor=120.0,
    sigma_r=15.0,
    sigma_f=35.0,
    z0=0.0,
    ellipticity=1.0,
    azimuth=0.0,
    trend=(0.0, 0.0),
    noise_std=0.0,
    seed=None,
):
    """Genera un DEM sintético con un cráter de parámetros conocidos.

    Es el banco de pruebas de la Fase 0: si el scoring no recupera estos
    parámetros, no tiene sentido apuntarlo a un DEM real.

    ellipticity : razón eje mayor / eje menor (1.0 = circular)
    azimuth     : orientación del eje mayor [rad]
    trend       : (dz/dx, dz/dy) pendiente regional añadida [m/px]
    noise_std   : desviación estándar de ruido gaussiano blanco [m]
    """
    ny, nx = shape
    if center is None:
        center = (nx / 2.0, ny / 2.0)
    cx, cy = center

    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    dx, dy = xx - cx, yy - cy

    ca, sa = np.cos(azimuth), np.sin(azimuth)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    r = np.sqrt((u / ellipticity) ** 2 + v ** 2)

    dem = crater_profile(r, z0, dh_rim, r_rim, sigma_r, dh_floor, sigma_f)
    dem = dem + trend[0] * dx + trend[1] * dy

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        dem = dem + rng.normal(0.0, noise_std, size=dem.shape)
    return dem


# ----------------------------------------------------------------------
# Preprocesamiento
# ----------------------------------------------------------------------
def detrend_plane(dem):
    """[C5] Resta el plano de mínimos cuadrados. Devuelve (dem_detrended, coef).

    Sin esto, el R^2 del ajuste del perfil mide sobre todo la pendiente
    regional y no la morfología del candidato.
    """
    dem = np.asarray(dem, dtype=float)
    ny, nx = dem.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    good = np.isfinite(dem)
    if good.sum() < 3:
        return dem.copy(), np.zeros(3)
    A = np.column_stack([xx[good], yy[good], np.ones(good.sum())])
    coef, *_ = np.linalg.lstsq(A, dem[good], rcond=None)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    return dem - plane, coef


# ----------------------------------------------------------------------
# Perfiles radiales
# ----------------------------------------------------------------------
def radial_profiles(dem, cx, cy, r_max, n_az=36, dr=1.0):
    """[C4] Extrae n_az perfiles radiales por interpolación bilineal.

    Devuelve (r, prof) con r de forma (n_r,) y prof de forma (n_az, n_r).
    Los puntos fuera de la ventana o sobre NaN quedan como NaN.
    """
    dem = np.asarray(dem, dtype=float)
    r = np.arange(dr, r_max + dr, dr)
    ang = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)

    xs = cx + np.outer(np.cos(ang), r)
    ys = cy + np.outer(np.sin(ang), r)

    filled = np.where(np.isfinite(dem), dem, 0.0)
    valid = np.isfinite(dem).astype(float)

    vals = map_coordinates(filled, [ys.ravel(), xs.ravel()], order=1,
                           mode="constant", cval=np.nan)
    wts = map_coordinates(valid, [ys.ravel(), xs.ravel()], order=1,
                          mode="constant", cval=0.0)

    prof = np.where(wts > 0.999, vals, np.nan).reshape(xs.shape)

    inside = ((xs >= 0) & (xs <= dem.shape[1] - 1) &
              (ys >= 0) & (ys <= dem.shape[0] - 1))
    prof[~inside] = np.nan
    return r, prof


def fit_radial_profile(r, z, p0=None):
    """Ajusta `crater_profile` al perfil radial medio, con cotas que imponen
    la geometría de cráter (dh_rim >= 0, dh_floor >= 0).

    Que las cotas fuercen un cráter es deliberado: el R^2 resultante mide
    cuánto se parece el candidato a un cráter, no cuán bien se ajusta
    cualquier curva. Una estructura invertida da R^2 bajo, que es lo que
    queremos.

    Devuelve (popt, r2). popt = (z0, dh_rim, r_rim, sigma_r, dh_floor, sigma_f).
    Si el ajuste no converge devuelve (None, 0.0).
    """
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    good = np.isfinite(z)
    if good.sum() < 12:
        return None, 0.0
    r, z = r[good], z[good]

    relief = float(np.nanmax(z) - np.nanmin(z))
    if relief < _EPS:
        return None, 0.0
    r_max = float(r.max())

    if p0 is None:
        r_guess = float(r[int(np.argmax(z))])
        r_guess = float(np.clip(r_guess, 0.15 * r_max, 0.85 * r_max))
        p0 = [float(np.median(z)),
              max(relief * 0.3, _EPS),
              r_guess,
              max(0.15 * r_guess, 1.0),
              max(relief * 0.5, _EPS),
              max(0.40 * r_guess, 1.0)]

    lo = [np.min(z) - 5 * relief, 0.0, 0.10 * r_max, 0.5, 0.0, 0.5]
    hi = [np.max(z) + 5 * relief, 10 * relief, 0.95 * r_max, r_max,
          10 * relief, r_max]
    p0 = [float(np.clip(v, l + _EPS, h - _EPS)) for v, l, h in zip(p0, lo, hi)]

    try:
        popt, _ = curve_fit(crater_profile, r, z, p0=p0,
                            bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError):
        return None, 0.0

    resid = z - crater_profile(r, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0
    return popt, float(np.clip(r2, 0.0, 1.0))


def azimuthal_coherence(r, prof, r_rim, band=0.5):
    """Consistencia azimutal de la posición del borde.

    Para cada azimut se busca un máximo de elevación ESTRICTAMENTE INTERIOR a
    la banda [r_rim*(1-band), r_rim*(1+band)]. Un azimut cuyo máximo cae en el
    extremo de la banda no ha detectado un borde: ha detectado una pendiente
    monótona. Ese requisito es lo que separa un anillo cerrado de un valle
    lineal o una ladera, que de otro modo ajustan el modelo de cráter con
    R^2 alto (comprobado: un valle gaussiano recto da R^2 = 1.00).

    coherencia = (fracción de azimuts con máximo interior)
                 x (1 - dispersión de esos radios / (0.5 * r_rim))

    Devuelve un valor en [0, 1].
    """
    r = np.asarray(r, float)
    prof = np.asarray(prof, float)
    if not np.isfinite(r_rim) or r_rim <= 0:
        return 0.0

    lo, hi = r_rim * (1.0 - band), r_rim * (1.0 + band)
    sel = (r >= lo) & (r <= hi)
    if sel.sum() < 5:
        return 0.0

    rs, zs = r[sel], prof[:, sel]
    n_band = rs.size
    peaks, n_used = [], 0
    for row in zs:
        if np.isfinite(row).sum() < 5:
            continue
        n_used += 1
        k = int(np.nanargmax(row))
        if k == 0 or k == n_band - 1:
            continue                      # máximo en el borde: no es un anillo
        peaks.append(rs[k])

    if n_used < 8 or len(peaks) < 4:
        return 0.0

    frac = len(peaks) / n_used
    peaks = np.asarray(peaks, float)
    spread = 1.4826 * float(np.median(np.abs(peaks - np.median(peaks))))
    tight = np.clip(1.0 - spread / (0.5 * r_rim), 0.0, 1.0)
    return float(np.clip(frac * tight, 0.0, 1.0))


def ring_contrast(dem, cx, cy, r_rim, dr=None):
    """Ec. 4 del documento: media en el anillo menos media en el disco interno,
    normalizada por la desviación estándar de la ventana (adimensional).

    Positivo => borde elevado respecto al interior (signatura de cráter).
    """
    dem = np.asarray(dem, float)
    if not np.isfinite(r_rim) or r_rim <= 0:
        return 0.0
    if dr is None:
        dr = max(0.15 * r_rim, 1.0)

    ny, nx = dem.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    rr = np.hypot(xx - cx, yy - cy)

    ring = (rr >= r_rim - dr) & (rr <= r_rim + dr) & np.isfinite(dem)
    inner = (rr <= 0.5 * r_rim) & np.isfinite(dem)
    if ring.sum() < 10 or inner.sum() < 10:
        return 0.0

    sd = float(np.nanstd(dem))
    if sd < _EPS:
        return 0.0
    return float((dem[ring].mean() - dem[inner].mean()) / sd)


# ----------------------------------------------------------------------
# Circularidad
# ----------------------------------------------------------------------
def sublevel_basin_mask(dem, cx, cy, r_max=None, level=None, quantile=0.5,
                        smooth_sigma=None):
    """[C2] Máscara de la depresión conectada que contiene el centro.

    Umbraliza el DEM en su cuantil `quantile` y devuelve la componente conexa
    que incluye (cx, cy). Reemplaza la ventana rectangular del borrador, que
    hacía que el BCI fuera constante.

    Nota: esto NO es una cuenca de drenaje. Es una máscara de sublevel set.
    Si el documento va a hablar de "cuenca de drenaje" hay que implementar
    delineación hidrológica de verdad (Priority-Flood + acumulación de flujo);
    mientras tanto, el texto debe decir "sublevel set", que es lo que hace
    el código.
    """
    dem = np.asarray(dem, float)

    # Suavizado previo. Sin él, el contorno del sublevel set de un cráter
    # degradado (borde de 8 m sobre ruido de 4 m) es dentado, el perímetro
    # se dispara y el BCI colapsa a ~0.05 aunque la depresión sea circular.
    if smooth_sigma is None and r_max is not None and np.isfinite(r_max):
        smooth_sigma = max(2.0, 0.04 * r_max)
    if smooth_sigma:
        ok = np.isfinite(dem).astype(float)
        filled = np.where(np.isfinite(dem), dem, 0.0)
        num = gaussian_filter(filled, smooth_sigma)
        den = gaussian_filter(ok, smooth_sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            dem = np.where(den > 1e-6, num / den, np.nan)

    good = np.isfinite(dem)
    if r_max is not None and np.isfinite(r_max) and r_max > 0:
        ny, nx = dem.shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        good = good & (np.hypot(xx - cx, yy - cy) <= r_max)
    if good.sum() < 10:
        return np.zeros_like(dem, dtype=bool)

    thr = float(level) if level is not None else float(
        np.quantile(dem[good], quantile))
    binary = good & (dem <= thr)
    lab, n = label(binary)
    if n == 0:
        return np.zeros_like(dem, dtype=bool)

    iy, ix = int(round(cy)), int(round(cx))
    iy = int(np.clip(iy, 0, dem.shape[0] - 1))
    ix = int(np.clip(ix, 0, dem.shape[1] - 1))
    comp = lab[iy, ix]
    if comp == 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        comp = int(np.argmax(sizes))
    mask = lab == comp
    mask = binary_fill_holes(mask)
    mask = binary_opening(mask, np.ones((3, 3), bool))
    return mask


def basin_circularity(mask, method="contour"):
    """Índice de circularidad de Miller: BCI = 4*pi*A / P^2.

    method="contour" : [C3] perímetro del contorno vectorizado (marching
                       squares). OJO: la digitalización sobrestima el
                       perímetro en ~6%, así que un disco rasterizado perfecto
                       da BCI = 0.89, no 1.00. Ese es el techo real de la
                       métrica y cualquier umbral debe fijarse contra él.
    method="pixel"   : perímetro por conteo de píxeles de borde (4-conexo).
                       SUBESTIMA el perímetro de un disco en ~10% y da
                       BCI = 1.24, un valor geométricamente imposible.
                       No usar para umbrales.
    method="borrador": reproduce la fórmula literal del código v2.0, cuya
                       precedencia de operadores (& antes que |) cuenta solo
                       los bordes inferior y derecho. Subestima el perímetro
                       en ~46% e INFLA el BCI de un disco a 3.4. Se conserva
                       solo como prueba de regresión del bug.
    """
    mask = np.asarray(mask, dtype=bool)
    area = float(mask.sum())
    if area < 8:
        return 0.0

    if method == "borrador":
        perim = float(np.sum(
            (mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)) |
            (mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False))))
    elif method == "pixel":
        up = np.zeros_like(mask); up[:-1, :] = mask[1:, :]
        left = np.zeros_like(mask); left[:, :-1] = mask[:, 1:]
        down = np.zeros_like(mask); down[1:, :] = mask[:-1, :]
        right = np.zeros_like(mask); right[:, 1:] = mask[:, :-1]
        border = mask & ~(up & left & down & right)
        perim = float(border.sum())
    elif method == "contour":
        padded = np.pad(mask.astype(float), 1)
        contours = find_contours(padded, 0.5)
        if not contours:
            return 0.0
        cont = max(contours, key=len)
        d = np.diff(cont, axis=0)
        perim = float(np.hypot(d[:, 0], d[:, 1]).sum())
    else:
        raise ValueError(f"method desconocido: {method!r}")

    if perim < _EPS:
        return 0.0
    return float(4.0 * np.pi * area / perim ** 2)


# ----------------------------------------------------------------------
# Interfaz de alto nivel
# ----------------------------------------------------------------------
def morphometric_metrics(dem, cx=None, cy=None, r_max=None, n_az=36,
                         detrend=True):
    """Calcula todas las métricas morfométricas de un candidato.

    Devuelve un dict con métricas CRUDAS. Deliberadamente NO devuelve un
    score compuesto: los pesos w1..w4 del borrador no están calibrados por
    nada, y con ~8 positivos no se pueden calibrar. La combinación es una
    decisión posterior, declarada y sujeta a análisis de sensibilidad.

    Claves:
      r2               bondad de ajuste del perfil radial medio [0,1]
      r_rim_px         radio del borde ajustado [px]
      dh_rim, dh_floor amplitudes ajustadas [unidades del DEM]
      coherence        consistencia azimutal del borde [0,1]
      ring_contrast    contraste anillo/interior, normalizado (sigma)
      bci              circularidad de Miller de la depresión conectada
      basin_area_px    área de esa depresión [px]
      relief           relieve de la ventana (tras detrending si aplica)
      edge_limited     True si r_rim > 0.8*r_max: la estructura ocupa casi
                       toda la ventana y el terreno no alcanza a volver al
                       nivel de fondo. En ese caso el ajuste NO es una
                       detección: es el borde de la ventana. Un valle lineal
                       recto ajusta el modelo con R^2 = 1.00 en esta
                       condición. La ventana debe ser >= 2.5 veces el
                       diámetro esperado.
    """
    dem = np.asarray(dem, dtype=float)
    ny, nx = dem.shape
    if cx is None:
        cx = (nx - 1) / 2.0
    if cy is None:
        cy = (ny - 1) / 2.0
    if r_max is None:
        r_max = 0.45 * min(nx, ny)

    work, _ = detrend_plane(dem) if detrend else (dem.copy(), None)

    r, prof = radial_profiles(work, cx, cy, r_max, n_az=n_az)
    with np.errstate(invalid="ignore"):
        mean_prof = np.nanmean(prof, axis=0)

    popt, r2 = fit_radial_profile(r, mean_prof)
    if popt is None:
        return {"r2": 0.0, "r_rim_px": np.nan, "dh_rim": np.nan,
                "dh_floor": np.nan, "coherence": 0.0, "ring_contrast": 0.0,
                "bci": 0.0, "basin_area_px": 0.0,
                "relief": float(np.nanmax(work) - np.nanmin(work)),
                "edge_limited": True}

    _, dh_rim, r_rim, _, dh_floor, _ = popt
    z0_fit = float(popt[0])
    mask = sublevel_basin_mask(work, cx, cy, r_max=1.2 * r_rim, level=z0_fit)

    return {
        "r2": r2,
        "r_rim_px": float(r_rim),
        "dh_rim": float(dh_rim),
        "dh_floor": float(dh_floor),
        "coherence": azimuthal_coherence(r, prof, r_rim),
        "ring_contrast": ring_contrast(work, cx, cy, r_rim),
        "bci": basin_circularity(mask, method="contour"),
        "basin_area_px": float(mask.sum()),
        "relief": float(np.nanmax(work) - np.nanmin(work)),
        "edge_limited": bool(r_rim > 0.8 * r_max),
    }
