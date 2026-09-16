"""
test_gee_tiles.py — Pruebas del paso 2, TODAS sin red y sin GEE.

La descarga en sí no se puede probar sin credenciales, pero todo lo que la
rodea sí: la zona UTM, el tamaño de la ventana, la georreferenciación, la
lectura del GeoTIFF y la conversión de píxeles a kilómetros. Esas son las
partes donde de verdad se cometen errores silenciosos.

Ejecutar:  pytest -q test_gee_tiles.py
"""

import os

import numpy as np
import pytest

from gee_tiles import (
    TileSpec,
    analyze_tile,
    load_tile,
    tile_bounds_utm,
    utm_epsg,
    window_size_px,
    write_synthetic_geotiff,
)


# ----------------------------------------------------------------------
# Zona UTM
# ----------------------------------------------------------------------
@pytest.mark.parametrize("lon,lat,esperado", [
    (-46.87, -8.08, "EPSG:32723"),    # Serra da Cangalha (sur)
    (-69.25, 4.50, "EPSG:32619"),     # Estructura del Vichada (norte)
    (-52.99, -16.79, "EPSG:32722"),   # Araguainha
    (-76.53, 3.42, "EPSG:32618"),     # Cali
])
def test_zona_utm(lon, lat, esperado):
    assert utm_epsg(lon, lat) == esperado


def test_hemisferio_cambia_el_codigo():
    """Mismo meridiano, distinto hemisferio: 326xx vs 327xx."""
    assert utm_epsg(-69.25, 4.5)[:8] == "EPSG:326"
    assert utm_epsg(-69.25, -4.5)[:8] == "EPSG:327"


def test_coordenadas_invalidas():
    with pytest.raises(ValueError):
        utm_epsg(200.0, 0.0)


# ----------------------------------------------------------------------
# Tamaño de ventana: la regla de 2.5 diámetros
# ----------------------------------------------------------------------
def test_ventana_cubre_dos_y_medio_diametros():
    n = window_size_px(13.0, pixel_m=30.0, factor=2.5)
    assert n * 30.0 >= 2.5 * 13000.0
    assert n % 2 == 0, "el lado debe ser par para tener centro definido"


def test_ventana_escala_con_el_diametro():
    assert window_size_px(50.0) > window_size_px(13.0) > window_size_px(4.5)


def test_ventana_rechaza_valores_no_positivos():
    with pytest.raises(ValueError):
        window_size_px(0.0)


def test_radio_esperado_no_dispara_edge_limited():
    """Con factor 2.5, r_rim queda en 0.44*r_max: lejos del umbral de 0.8.

    Si alguien baja `factor`, esta prueba avisa antes de que las métricas
    empiecen a marcar todo como limitado por la ventana.
    """
    spec = TileSpec("X", -46.87, -8.08, 13.0, "FABDEM")
    r_max = 0.45 * spec.n_px
    assert spec.expected_r_rim_px / r_max < 0.8


# ----------------------------------------------------------------------
# Georreferenciación
# ----------------------------------------------------------------------
def test_bounds_centrados_y_cuadrados():
    crs, (xmin, ymin, xmax, ymax), (x, y) = tile_bounds_utm(-46.87, -8.08, 16260.0)
    assert crs == "EPSG:32723"
    assert (xmax - xmin) == pytest.approx(ymax - ymin)
    assert (xmin + xmax) / 2 == pytest.approx(x)
    assert (ymin + ymax) / 2 == pytest.approx(y)


def test_nombre_de_archivo_es_seguro():
    spec = TileSpec("Serra da Cangalha", -46.87, -8.08, 13.0, "GLO30")
    fn = spec.filename()
    assert " " not in fn and fn.endswith(".tif")
    assert "GLO30" in fn


# ----------------------------------------------------------------------
# Ida y vuelta completa: escribir GeoTIFF -> leer -> medir
# ----------------------------------------------------------------------
@pytest.fixture
def tesela_sintetica(tmp_path):
    spec = TileSpec("Prueba", -46.87, -8.08, 13.0, "FABDEM")
    path = write_synthetic_geotiff(str(tmp_path / "t.tif"), spec)
    return spec, path


def test_geotiff_tiene_pixeles_cuadrados_en_metros(tesela_sintetica):
    _, path = tesela_sintetica
    _, info = load_tile(path)
    assert info["pixel_x_m"] == pytest.approx(30.0)
    assert info["pixel_y_m"] == pytest.approx(30.0)
    assert info["crs"] == "EPSG:32723"


def test_recupera_el_diametro_del_catalogo(tesela_sintetica):
    """La prueba que importa: de coordenada geográfica a diámetro en km."""
    spec, path = tesela_sintetica
    m = analyze_tile(path, spec)
    assert m["error_relativo"] < 0.15
    assert m["r2"] > 0.90
    assert m["coherence"] > 0.80
    assert not m["edge_limited"]


def test_nodata_se_convierte_en_nan(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    arr = np.full((64, 64), 100.0, dtype="float32")
    arr[10:20, 10:20] = -9999.0
    with rasterio.open(tmp_path / "n.tif", "w", driver="GTiff", height=64,
                       width=64, count=1, dtype="float32", crs="EPSG:32723",
                       transform=from_origin(0, 0, 30, 30),
                       nodata=-9999.0) as dst:
        dst.write(arr, 1)
    dem, info = load_tile(str(tmp_path / "n.tif"))
    assert np.isnan(dem[15, 15])
    assert info["nan_frac"] == pytest.approx(100 / 4096, abs=1e-6)


def test_geotiff_sin_nodata_declarado_no_rompe(tmp_path):
    """El código v2.0 hacía `dem[dem == src.nodata]`, que falla si es None."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(tmp_path / "s.tif", "w", driver="GTiff", height=32,
                       width=32, count=1, dtype="float32", crs="EPSG:32723",
                       transform=from_origin(0, 0, 30, 30)) as dst:
        dst.write(np.zeros((32, 32), dtype="float32"), 1)
    dem, info = load_tile(str(tmp_path / "s.tif"))
    assert info["nodata"] is None
    assert np.isfinite(dem).all()
