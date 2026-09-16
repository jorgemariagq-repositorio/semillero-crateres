"""
gee_tiles.py — Extracción de teselas de DEM desde Google Earth Engine.

Paso 2 del proyecto. Complementa a `craterscore.py`: este módulo se encarga
de conseguir el dato; aquel de medirlo.

DECISIONES DE DISEÑO, y por qué

  [D1] Teselas, no mosaicos. Cada objetivo se descarga como una ventana
       independiente de unos pocos MB. Exportar el territorio completo a 30 m
       produce un GeoTIFF de ~5 GB y pide >20 GB de RAM al analizarlo: no
       corre en Colab ni en un portátil.

  [D2] Reproyección a UTM local. En EPSG:4326 el píxel mide 30 m en latitud
       pero 30*cos(lat) m en longitud. A 8 S eso es una anisotropía del 1%,
       que basta para sesgar el índice de circularidad y el radio del borde.
       Todas las teselas se entregan en UTM, con píxeles cuadrados en metros.

  [D3] Ventana = 2.5 x diámetro esperado. Es la regla operativa que sale de
       la Fase 0: por debajo de eso la estructura ocupa casi toda la ventana,
       el terreno no vuelve al nivel de fondo y `edge_limited` se dispara.
       Un valle recto ajusta el modelo de cráter con R2 = 1.00 en esa
       condición.

  [D4] Remuestreo bilineal antes de reproyectar. El vecino más próximo
       introduce escalones de altura que contaminan las derivadas y el
       perímetro del sublevel set.

  [D5] Caché en disco. Las sesiones de Colab se caen. Si el archivo existe y
       `overwrite=False`, no se vuelve a descargar.

REQUISITOS
  pip install earthengine-api rasterio pyproj requests
  Cuenta de GEE con un proyecto de Google Cloud asociado (obligatorio desde
  2023). `ee.Initialize(project="mi-proyecto")` falla sin él.

FABDEM se distribuye bajo CC BY-NC-SA 4.0. Uso no comercial y atribución
obligatoria: "FABDEM is produced using Copernicus WorldDEM-30 (c) DLR e.V."
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict

import numpy as np

__all__ = [
    "DATASETS",
    "utm_epsg",
    "window_size_px",
    "tile_bounds_utm",
    "TileSpec",
    "fetch_tile",
    "load_tile",
    "analyze_tile",
    "write_synthetic_geotiff",
]

# ----------------------------------------------------------------------
# Catálogo de fuentes
# ----------------------------------------------------------------------
#: Cada entrada: (id de la colección, banda de elevación, tipo de modelo).
#: GLO30 es un DSM (mide el techo del dosel). FABDEM le quita árboles y
#: edificios. La comparación entre ambos es el Objetivo Específico 2.
DATASETS = {
    "FABDEM": {
        "collection": "projects/sat-io/open-datasets/FABDEM",
        "band": None,          # la colección trae una sola banda sin nombrar
        "kind": "DTM",
        "license": "CC BY-NC-SA 4.0",
    },
    "GLO30": {
        "collection": "COPERNICUS/DEM/GLO30",
        "band": "DEM",
        "kind": "DSM",
        "license": "Copernicus, uso libre con atribución",
    },
}


# ----------------------------------------------------------------------
# Geometría de la tesela (todo esto es offline y testeable sin GEE)
# ----------------------------------------------------------------------
def utm_epsg(lon: float, lat: float) -> str:
    """Código EPSG de la zona UTM que contiene (lon, lat).

    >>> utm_epsg(-46.87, -8.08)      # Serra da Cangalha, Brasil
    'EPSG:32723'
    >>> utm_epsg(-69.25, 4.50)       # Estructura del Vichada, Colombia
    'EPSG:32619'
    """
    if not -180.0 <= lon <= 180.0 or not -80.0 <= lat <= 84.0:
        raise ValueError(f"coordenadas fuera del dominio UTM: {lon}, {lat}")
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def window_size_px(diameter_km: float, pixel_m: float = 30.0,
                   factor: float = 2.5) -> int:
    """Lado de la ventana en píxeles. [D3]

    factor = 2.5 es la regla operativa de la Fase 0. No bajarlo sin repetir
    la prueba del valle lineal.

    >>> window_size_px(13.0)         # Serra da Cangalha, 13 km
    1084
    """
    if diameter_km <= 0 or pixel_m <= 0 or factor <= 0:
        raise ValueError("diameter_km, pixel_m y factor deben ser positivos")
    n = int(math.ceil(diameter_km * 1000.0 * factor / pixel_m))
    return n + (n % 2)          # par, para que haya un centro bien definido


def tile_bounds_utm(lon: float, lat: float, half_width_m: float,
                    crs: str | None = None):
    """Rectángulo (xmin, ymin, xmax, ymax) en metros UTM centrado en (lon, lat).

    Devuelve (crs, bounds, (easting, northing) del centro).
    """
    from pyproj import Transformer

    crs = crs or utm_epsg(lon, lat)
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = fwd.transform(lon, lat)
    bounds = (x - half_width_m, y - half_width_m,
              x + half_width_m, y + half_width_m)
    return crs, bounds, (x, y)


@dataclass
class TileSpec:
    """Todo lo que define una tesela. Se serializa junto al GeoTIFF."""
    name: str
    lon: float
    lat: float
    diameter_km: float
    dataset: str
    pixel_m: float = 30.0
    factor: float = 2.5

    @property
    def n_px(self) -> int:
        return window_size_px(self.diameter_km, self.pixel_m, self.factor)

    @property
    def half_width_m(self) -> float:
        return self.n_px * self.pixel_m / 2.0

    @property
    def crs(self) -> str:
        return utm_epsg(self.lon, self.lat)

    @property
    def expected_r_rim_px(self) -> float:
        """Radio del borde esperado, en píxeles. Sirve para comprobar que la
        compuerta recupere algo compatible con el catálogo."""
        return self.diameter_km * 1000.0 / 2.0 / self.pixel_m

    def filename(self) -> str:
        slug = "".join(c if c.isalnum() else "_" for c in self.name).strip("_")
        return f"{slug}__{self.dataset}__{self.pixel_m:.0f}m.tif"


# ----------------------------------------------------------------------
# Descarga desde GEE
# ----------------------------------------------------------------------
def _ee_image(dataset: str):
    """Imagen mosaico de la fuente pedida, con remuestreo bilineal. [D4]"""
    import ee

    if dataset not in DATASETS:
        raise KeyError(f"dataset desconocido: {dataset!r}. "
                       f"Opciones: {list(DATASETS)}")
    cfg = DATASETS[dataset]
    col = ee.ImageCollection(cfg["collection"])
    if cfg["band"]:
        col = col.select(cfg["band"])
    return col.mosaic().rename("elevation").resample("bilinear")


def fetch_tile(spec: TileSpec, out_dir: str = "tiles",
               overwrite: bool = False, timeout: int = 300) -> str:
    """Descarga la tesela y devuelve la ruta del GeoTIFF. [D1] [D2] [D5]

    Requiere `ee.Initialize(project=...)` ya ejecutado.
    """
    import ee
    import requests

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, spec.filename())
    if os.path.exists(path) and not overwrite:
        print(f"[cache] {path}")
        return path

    crs, bounds, _ = tile_bounds_utm(spec.lon, spec.lat, spec.half_width_m)
    region = ee.Geometry.Rectangle(list(bounds), proj=crs, geodesic=False)

    url = _ee_image(spec.dataset).getDownloadURL({
        "region": region,
        "crs": crs,
        "scale": spec.pixel_m,
        "format": "GEO_TIFF",
    })

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with open(path, "wb") as fh:
        fh.write(resp.content)
    print(f"[ok] {path}  ({len(resp.content) / 1e6:.1f} MB, {crs}, "
          f"{spec.n_px}x{spec.n_px} px nominales)")
    return path


# ----------------------------------------------------------------------
# Lectura y control de calidad
# ----------------------------------------------------------------------
def load_tile(path: str):
    """Lee el GeoTIFF. Devuelve (dem, info).

    `dem` es float64 con los nodata convertidos a NaN. `info` trae CRS, tamaño
    de píxel en x y en y, y fracción de NaN. Revísenlo siempre: una tesela con
    30% de NaN no sirve y hay que saberlo antes de interpretar métricas.
    """
    import rasterio

    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
        tr = src.transform
        info = {
            "path": path,
            "crs": str(src.crs),
            "shape": (src.height, src.width),
            "pixel_x_m": abs(tr.a),
            "pixel_y_m": abs(tr.e),
            "nodata": nodata,
        }

    # CAMBIO respecto al código v2.0: `dem == src.nodata` reventaba cuando
    # nodata era None, y no capturaba los NaN ya presentes en el archivo.
    if nodata is not None and np.isfinite(nodata):
        dem = np.where(np.isclose(dem, nodata), np.nan, dem)
    dem = np.where(np.abs(dem) > 1e5, np.nan, dem)   # centinelas tipo -9999
    info["nan_frac"] = float(np.mean(~np.isfinite(dem)))

    aniso = abs(info["pixel_x_m"] - info["pixel_y_m"]) / info["pixel_x_m"]
    if aniso > 0.01:
        print(f"[AVISO] píxeles no cuadrados ({aniso:.1%}). ¿Se reproyectó "
              f"a UTM? El BCI y el radio del borde quedan sesgados.")
    if info["nan_frac"] > 0.05:
        print(f"[AVISO] {info['nan_frac']:.1%} de la tesela es NaN.")
    return dem, info


def analyze_tile(path: str, spec: TileSpec | None = None,
                 r_max_frac: float = 0.45) -> dict:
    """Corre las métricas morfométricas sobre una tesela descargada.

    El centro de análisis es el centro geométrico de la ventana, que por
    construcción es la coordenada del objetivo. Devuelve las métricas de
    `craterscore` más la conversión a unidades físicas.
    """
    from craterscore import morphometric_metrics

    dem, info = load_tile(path)
    ny, nx = dem.shape
    px = info["pixel_x_m"]

    m = morphometric_metrics(
        dem,
        cx=(nx - 1) / 2.0,
        cy=(ny - 1) / 2.0,
        r_max=r_max_frac * min(nx, ny),
    )
    m["diameter_km_medido"] = 2.0 * m["r_rim_px"] * px / 1000.0
    m.update({k: info[k] for k in ("crs", "shape", "pixel_x_m", "nan_frac")})

    if spec is not None:
        m["objetivo"] = spec.name
        m["dataset"] = spec.dataset
        m["diameter_km_catalogo"] = spec.diameter_km
        d_cat = spec.diameter_km
        m["error_relativo"] = (abs(m["diameter_km_medido"] - d_cat) / d_cat
                               if d_cat > 0 else float("nan"))
    return m


# ----------------------------------------------------------------------
# Autoprueba sin GEE
# ----------------------------------------------------------------------
def write_synthetic_geotiff(path: str, spec: TileSpec, noise_std: float = 3.0,
                            dh_rim: float = 60.0, dh_floor: float = 200.0,
                            seed: int = 20260906) -> str:
    """Escribe un GeoTIFF sintético con la georreferenciación real de `spec`.

    Sirve para validar todo el camino (georreferenciación, lectura, métricas,
    conversión a km) SIN autenticarse en GEE. Ejecútenlo antes de gastar
    cuota: si la autoprueba no recupera el diámetro del catálogo, el problema
    es del código y no de los datos.
    """
    import rasterio
    from rasterio.transform import from_origin

    from craterscore import synthetic_crater

    n = spec.n_px
    r_rim_px = spec.expected_r_rim_px
    dem = synthetic_crater(
        shape=(n, n), r_rim=r_rim_px,
        dh_rim=dh_rim, dh_floor=dh_floor,
        sigma_r=0.22 * r_rim_px, sigma_f=0.55 * r_rim_px,
        z0=250.0, noise_std=noise_std, seed=seed,
    )

    crs, bounds, _ = tile_bounds_utm(spec.lon, spec.lat, spec.half_width_m)
    transform = from_origin(bounds[0], bounds[3], spec.pixel_m, spec.pixel_m)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs=crs, transform=transform,
                       nodata=-9999.0, compress="deflate") as dst:
        dst.write(dem.astype("float32"), 1)
    return path


def selftest(verbose: bool = True) -> dict:
    """Autoprueba completa sin red. Devuelve las métricas obtenidas."""
    import tempfile

    spec = TileSpec(name="Autoprueba", lon=-46.87, lat=-8.08,
                    diameter_km=13.0, dataset="FABDEM")
    with tempfile.TemporaryDirectory() as tmp:
        path = write_synthetic_geotiff(os.path.join(tmp, "test.tif"), spec)
        m = analyze_tile(path, spec)
    if verbose:
        print(f"CRS            : {m['crs']}")
        print(f"Tamaño         : {m['shape'][0]}x{m['shape'][1]} px "
              f"@ {m['pixel_x_m']:.1f} m")
        print(f"Diám. catálogo : {m['diameter_km_catalogo']:.2f} km")
        print(f"Diám. medido   : {m['diameter_km_medido']:.2f} km "
              f"(error {m['error_relativo']:.1%})")
        print(f"R2 {m['r2']:.2f} | coherencia {m['coherence']:.2f} | "
              f"anillo {m['ring_contrast']:+.2f} | BCI {m['bci']:.2f} | "
              f"edge_limited {m['edge_limited']}")
    return m


if __name__ == "__main__":
    selftest()
