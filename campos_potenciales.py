"""
campos_potenciales.py — Analisis de anomalias de gravedad y magneticas.

Fase 3 del proyecto: Estructura del Vichada. Complementa a `craterscore.py`
(que mide forma del terreno) y a `gee_tiles.py` (que consigue el DEM).

POR QUE ESTE MODULO EXISTE
La Estructura del Vichada esta enterrada bajo cientos de metros de sedimento.
Su forma superficial dice poco. La evidencia que se ha invocado a favor del
origen de impacto es una anomalia gravimetrica, asi que la unica manera de
evaluarla es mirar campos potenciales.

QUE HACE Y QUE NO HACE
Hace: leer grillas de anomalia, separar regional de residual, y extraer
perfiles radiales alrededor de un punto. Son MEDICIONES.
No hace: interpretar. La interpretacion de campos potenciales excede la
formacion del semillero y requiere la asesoria externa prevista en el riesgo
R3 del anteproyecto. Este modulo produce numeros; que significan esos numeros
es otra conversacion.

FUENTES DE DATOS (ninguna requiere cuenta; ver la guia del Paso 3)
  Gravedad   ICGEM, servicio de calculo, archivo .gdf
  Magnetismo NOAA NCEI, EMAG2 v3, CSV global o GeoTIFF

Dependencias: numpy, scipy. rasterio solo si se leen GeoTIFF.
"""

from __future__ import annotations

import numpy as np
import warnings

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

__all__ = [
    "KM_POR_GRADO_LAT",
    "leer_gdf_icgem",
    "leer_csv_generico",
    "leer_geotiff_ventana",
    "Grilla",
    "grados_a_km",
    "perfil_radial",
    "separar_regional_residual",
    "resumen_anomalia",
]

#: Longitud de un grado de latitud, en km. Suficiente para una ventana de
#: pocos grados; no usar para geodesia de precision.
KM_POR_GRADO_LAT = 110.574
KM_POR_GRADO_LON_ECUADOR = 111.320

_EPS = 1e-12


# ----------------------------------------------------------------------
# Estructura de datos
# ----------------------------------------------------------------------
class Grilla:
    """Grilla regular en coordenadas geograficas.

    Atributos
    ---------
    lon, lat : vectores 1D crecientes, en grados
    valores  : matriz 2D de forma (len(lat), len(lon))
    unidad   : texto libre, p.ej. "mGal" o "nT"
    fuente   : de donde salio, para trazabilidad
    """

    def __init__(self, lon, lat, valores, unidad="", fuente=""):
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        valores = np.asarray(valores, float)
        if valores.shape != (lat.size, lon.size):
            raise ValueError(
                f"forma incoherente: valores {valores.shape} pero "
                f"lat={lat.size}, lon={lon.size}")
        self.lon, self.lat, self.valores = lon, lat, valores
        self.unidad, self.fuente = unidad, fuente

    def __repr__(self):
        return (f"<Grilla {self.valores.shape} "
                f"lon[{self.lon.min():.3f},{self.lon.max():.3f}] "
                f"lat[{self.lat.min():.3f},{self.lat.max():.3f}] "
                f"{self.unidad}>")

    @property
    def paso_grados(self):
        """(paso en lon, paso en lat), en grados."""
        dlon = np.median(np.diff(self.lon)) if self.lon.size > 1 else np.nan
        dlat = np.median(np.diff(self.lat)) if self.lat.size > 1 else np.nan
        return float(dlon), float(dlat)

    def resolucion_km(self):
        """Tamano de celda en km, a la latitud central."""
        dlon, dlat = self.paso_grados
        lat0 = float(np.mean(self.lat))
        return (abs(dlon) * KM_POR_GRADO_LON_ECUADOR * np.cos(np.radians(lat0)),
                abs(dlat) * KM_POR_GRADO_LAT)

    def interpolador(self):
        return RegularGridInterpolator(
            (self.lat, self.lon), self.valores,
            bounds_error=False, fill_value=np.nan)

    def recortar(self, lon0, lat0, medio_ancho_grados):
        """Subgrilla cuadrada centrada en (lon0, lat0)."""
        ml = medio_ancho_grados
        i = (self.lat >= lat0 - ml) & (self.lat <= lat0 + ml)
        j = (self.lon >= lon0 - ml) & (self.lon <= lon0 + ml)
        if i.sum() < 3 or j.sum() < 3:
            raise ValueError("el recorte deja menos de 3 filas o columnas: "
                             "revisa que el punto este dentro de la grilla")
        return Grilla(self.lon[j], self.lat[i],
                      self.valores[np.ix_(i, j)], self.unidad, self.fuente)


# ----------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------
def leer_gdf_icgem(path, columna_valor=2):
    """Lee un archivo .gdf del servicio de calculo de ICGEM.

    El formato tiene una cabecera de texto que termina en una linea que empieza
    con `end_of_head`, y despues columnas numericas. Por defecto ICGEM escribe
    longitud, latitud y el funcional pedido, en ese orden.

    `columna_valor` es el indice (base 0) de la columna del funcional. Si el
    archivo trae mas de tres columnas, abrilo con un editor de texto y contalas
    antes de asumir nada.
    """
    lons, lats, vals = [], [], []
    en_datos = False
    unidad = ""

    with open(path, "r", errors="ignore") as fh:
        for linea in fh:
            if not en_datos:
                if "unit" in linea.lower() and not unidad:
                    partes = linea.split()
                    if len(partes) >= 2:
                        unidad = partes[-1]
                if linea.lstrip().startswith("end_of_head"):
                    en_datos = True
                continue
            partes = linea.split()
            if len(partes) <= columna_valor:
                continue
            try:
                lo, la = float(partes[0]), float(partes[1])
                v = float(partes[columna_valor])
            except ValueError:
                continue
            lons.append(lo); lats.append(la); vals.append(v)

    if not lons:
        raise ValueError(
            f"{path}: no se leyo ningun dato. Comprueba que el archivo tenga "
            f"una linea 'end_of_head' y al menos {columna_valor + 1} columnas.")

    return _puntos_a_grilla(lons, lats, vals, unidad or "mGal",
                            fuente=f"ICGEM gdf: {path}")


def leer_csv_generico(path, col_lon=0, col_lat=1, col_val=2, separador=None,
                      saltar=0, unidad="", filtro_lon=None, filtro_lat=None):
    """Lee una grilla desde un archivo de texto con columnas.

    Pensado para el CSV global de EMAG2 v3, que tiene mas de 50 millones de
    filas: por eso acepta `filtro_lon` y `filtro_lat` como pares (min, max) y
    descarta el resto mientras lee, sin cargar todo en memoria.

    ATENCION: el orden de las columnas de EMAG2 hay que leerlo en el archivo
    `EMAG2_readme.txt` que NOAA distribuye junto al dato. No lo asumas. Si
    pasas los indices equivocados vas a obtener una grilla que parece
    razonable y esta mal.
    """
    lons, lats, vals = [], [], []
    ncol = max(col_lon, col_lat, col_val) + 1

    with open(path, "r", errors="ignore") as fh:
        for _ in range(saltar):
            fh.readline()
        for linea in fh:
            partes = linea.split(separador) if separador else linea.split()
            if len(partes) < ncol:
                continue
            try:
                lo = float(partes[col_lon])
                la = float(partes[col_lat])
                v = float(partes[col_val])
            except ValueError:
                continue
            if lo > 180.0:
                lo -= 360.0          # EMAG2 usa 0..360 en algunas versiones
            if filtro_lon and not (filtro_lon[0] <= lo <= filtro_lon[1]):
                continue
            if filtro_lat and not (filtro_lat[0] <= la <= filtro_lat[1]):
                continue
            lons.append(lo); lats.append(la); vals.append(v)

    if not lons:
        raise ValueError(
            f"{path}: ninguna fila paso los filtros. Revisa los indices de "
            f"columna y los rangos de longitud y latitud.")
    return _puntos_a_grilla(lons, lats, vals, unidad, fuente=f"CSV: {path}")


def leer_geotiff_ventana(path, lon0, lat0, medio_ancho_grados, unidad=""):
    """Lee solo una ventana de un GeoTIFF global. Requiere rasterio.

    Evita cargar en memoria un raster global de varios cientos de MB.
    """
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        ml = medio_ancho_grados
        ventana = from_bounds(lon0 - ml, lat0 - ml, lon0 + ml, lat0 + ml,
                              transform=src.transform)
        datos = src.read(1, window=ventana).astype(float)
        tr = src.window_transform(ventana)
        nodata = src.nodata

    if nodata is not None and np.isfinite(nodata):
        datos = np.where(np.isclose(datos, nodata), np.nan, datos)

    ny, nx = datos.shape
    if ny < 3 or nx < 3:
        raise ValueError("la ventana quedo demasiado pequena: revisa que el "
                         "punto este dentro del raster")

    lon = tr.c + tr.a * (np.arange(nx) + 0.5)
    lat = tr.f + tr.e * (np.arange(ny) + 0.5)
    if lat[0] > lat[-1]:                 # rasterio suele entregar norte arriba
        lat = lat[::-1]
        datos = datos[::-1, :]
    return Grilla(lon, lat, datos, unidad, fuente=f"GeoTIFF: {path}")


def _puntos_a_grilla(lons, lats, vals, unidad, fuente):
    """Convierte puntos dispersos de una grilla regular a matriz 2D."""
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    vals = np.asarray(vals, float)

    ulon = np.unique(np.round(lons, 6))
    ulat = np.unique(np.round(lats, 6))
    if ulon.size * ulat.size != lons.size:
        raise ValueError(
            f"los puntos no forman una grilla regular completa: "
            f"{ulon.size} x {ulat.size} = {ulon.size * ulat.size} esperados, "
            f"{lons.size} leidos. Comprueba que pediste una grilla y no una "
            f"lista de puntos.")

    matriz = np.full((ulat.size, ulon.size), np.nan)
    i = np.searchsorted(ulat, np.round(lats, 6))
    j = np.searchsorted(ulon, np.round(lons, 6))
    matriz[i, j] = vals
    return Grilla(ulon, ulat, matriz, unidad, fuente)


# ----------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------
def grados_a_km(dlon, dlat, lat0):
    """Convierte diferencias en grados a km, en un plano local.

    Aproximacion equirectangular alrededor de lat0. El error es inferior al
    0,1% para ventanas de un par de grados, que es lo que usamos.
    """
    x = np.asarray(dlon, float) * KM_POR_GRADO_LON_ECUADOR * np.cos(np.radians(lat0))
    y = np.asarray(dlat, float) * KM_POR_GRADO_LAT
    return x, y


def perfil_radial(grilla, lon0, lat0, r_max_km, n_az=36, paso_km=1.0):
    """Perfiles radiales de la anomalia alrededor de (lon0, lat0).

    Devuelve (r_km, perfiles, perfil_medio) con `perfiles` de forma
    (n_az, n_r). Misma logica que `radial_profiles` de craterscore, pero
    muestreando una grilla geografica en lugar de una matriz de pixeles.
    """
    r = np.arange(paso_km, r_max_km + paso_km, paso_km)
    ang = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)

    x = np.outer(np.cos(ang), r)      # km este
    y = np.outer(np.sin(ang), r)      # km norte

    dlon = x / (KM_POR_GRADO_LON_ECUADOR * np.cos(np.radians(lat0)))
    dlat = y / KM_POR_GRADO_LAT

    pts = np.column_stack([(lat0 + dlat).ravel(), (lon0 + dlon).ravel()])
    vals = grilla.interpolador()(pts).reshape(x.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medio = np.nanmean(vals, axis=0)
    return r, vals, medio


# ----------------------------------------------------------------------
# Separacion regional / residual
# ----------------------------------------------------------------------
def separar_regional_residual(grilla, longitud_onda_km=150.0):
    """Separa la anomalia en componente regional y residual.

    El regional se estima con un filtro gaussiano pasa-bajas; el residual es la
    diferencia. Es el metodo mas simple que existe y tiene un supuesto fuerte:
    que la fuente de interes es mas pequena que `longitud_onda_km`. Para una
    estructura de 50 km, un corte de 150 km es un punto de partida razonable.

    ESTO ES UNA DECISION DECLARADA, NO CALIBRADA. Repitan el analisis con
    varios cortes (100, 150, 200, 300 km) y reporten como cambia el resultado.
    Si la conclusion depende del corte elegido, la conclusion no existe.

    Devuelve (regional, residual) como dos objetos Grilla.
    """
    res_x_km, res_y_km = grilla.resolucion_km()
    sigma_x = longitud_onda_km / max(res_x_km, _EPS) / 2.0
    sigma_y = longitud_onda_km / max(res_y_km, _EPS) / 2.0
    if min(sigma_x, sigma_y) < 1.0:
        raise ValueError(
            f"la longitud de onda pedida ({longitud_onda_km} km) es demasiado "
            f"corta para la resolucion de la grilla "
            f"({res_x_km:.1f} x {res_y_km:.1f} km).")

    datos = grilla.valores
    ok = np.isfinite(datos).astype(float)
    lleno = np.where(np.isfinite(datos), datos, 0.0)
    num = gaussian_filter(lleno, (sigma_y, sigma_x), mode="nearest")
    den = gaussian_filter(ok, (sigma_y, sigma_x), mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        regional = np.where(den > 1e-6, num / den, np.nan)

    reg = Grilla(grilla.lon, grilla.lat, regional, grilla.unidad,
                 f"{grilla.fuente} | regional {longitud_onda_km:.0f} km")
    resi = Grilla(grilla.lon, grilla.lat, datos - regional, grilla.unidad,
                  f"{grilla.fuente} | residual {longitud_onda_km:.0f} km")
    return reg, resi


# ----------------------------------------------------------------------
# Descriptores
# ----------------------------------------------------------------------
def resumen_anomalia(r_km, perfil_medio, radio_estructura_km):
    """Descriptores CRUDOS del perfil radial medio. No interpreta nada.

    Devuelve un dict con:
      valor_centro      media del perfil dentro de 0.3 R
      valor_borde       media del perfil en el anillo 0.8-1.2 R
      valor_fuera       media del perfil mas alla de 1.5 R
      contraste_centro  centro menos fuera
      contraste_borde   borde menos fuera
      signo             "negativa" si el centro esta por debajo del entorno
      amplitud_pico_a_pico
    """
    r = np.asarray(r_km, float)
    z = np.asarray(perfil_medio, float)
    R = float(radio_estructura_km)

    def _media(mascara):
        sel = z[mascara]
        sel = sel[np.isfinite(sel)]
        return float(np.mean(sel)) if sel.size else np.nan

    centro = _media(r <= 0.3 * R)
    borde = _media((r >= 0.8 * R) & (r <= 1.2 * R))
    fuera = _media(r >= 1.5 * R)

    finitos = z[np.isfinite(z)]
    return {
        "valor_centro": centro,
        "valor_borde": borde,
        "valor_fuera": fuera,
        "contraste_centro": centro - fuera,
        "contraste_borde": borde - fuera,
        "signo": ("negativa" if centro < fuera else "positiva"),
        "amplitud_pico_a_pico": (float(finitos.max() - finitos.min())
                                 if finitos.size else np.nan),
    }
