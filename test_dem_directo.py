"""
test_dem_directo.py — Pruebas de la ruta sin Google Earth Engine.

Todas sin red. Se fabrican teselas de 1 grado sinteticas con un crater de
parametros conocidos repartido entre ellas, y se comprueba que el modulo las
una, las reproyecte a UTM, recorte la ventana y recupere el diametro.

Lo que NO se puede probar aqui: la descarga real. Los patrones de URL y de
nombres de archivo hay que verificarlos contra los repositorios la primera vez.

Ejecutar:  pytest -q test_dem_directo.py
"""

import os
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from craterscore import crater_profile
from dem_directo import (construir_tesela_fabdem, extraer_de_zip_fabdem,
                         mosaicar_y_reproyectar, nombre_tesela,
                         nombre_tif_fabdem, nombre_zip_fabdem,
                         teselas_necesarias, url_glo30)
from gee_tiles import analyze_tile, TileSpec

# Serra da Cangalha. PENDIENTE DE VERIFICAR contra Kenkmann (2021).
SERRA = dict(lon=-46.87, lat=-8.08, diametro_km=13.0)
VICHADA = dict(lon=-69.25, lat=4.50, diametro_km=50.0)


# ----------------------------------------------------------------------
# Nomenclatura
# ----------------------------------------------------------------------
@pytest.mark.parametrize("lat,lon,esperado", [
    (-9, -47, "S09W047"),
    (-8, -48, "S08W048"),
    (4, -70, "N04W070"),
    (0, 0, "N00E000"),
])
def test_nombre_tesela(lat, lon, esperado):
    assert nombre_tesela(lat, lon) == esperado


def test_zip_de_10_grados():
    assert nombre_zip_fabdem(-9, -47) == "S10W050-N00W040_FABDEM_V1-2.zip"
    assert nombre_zip_fabdem(4, -70) == "N00W070-N10W060_FABDEM_V1-2.zip"


def test_las_dos_ventanas_caen_en_un_solo_zip():
    """Conveniencia real: cada objetivo necesita bajar un unico ZIP."""
    for caso in (SERRA, VICHADA):
        _, lista = teselas_necesarias(**caso)
        assert len({nombre_zip_fabdem(la, lo) for la, lo in lista}) == 1


def test_url_glo30_es_https_y_termina_en_tif():
    u = url_glo30(-9, -47)
    assert u.startswith("https://")
    assert u.endswith(".tif")
    assert "S09_00_W047_00" in u


# ----------------------------------------------------------------------
# Que teselas hacen falta
# ----------------------------------------------------------------------
def test_serra_necesita_cuatro_teselas():
    """La ventana cae justo sobre un cruce de teselas. No es una sola."""
    _, lista = teselas_necesarias(**SERRA)
    nombres = {nombre_tesela(la, lo) for la, lo in lista}
    assert nombres == {"S09W048", "S09W047", "S08W048", "S08W047"}


def test_vichada_necesita_seis_teselas():
    _, lista = teselas_necesarias(**VICHADA)
    nombres = {nombre_tesela(la, lo) for la, lo in lista}
    assert nombres == {"N03W070", "N03W069", "N04W070",
                       "N04W069", "N05W070", "N05W069"}


def test_la_ventana_respeta_el_factor_2_5():
    (lon0, lon1, lat0, lat1), _ = teselas_necesarias(**SERRA)
    ancho_km = (lat1 - lat0) * 110.574
    assert ancho_km == pytest.approx(2.5 * 13.0, rel=0.02)


def test_diametro_invalido():
    with pytest.raises(ValueError):
        teselas_necesarias(-46.87, -8.08, 0.0)


# ----------------------------------------------------------------------
# Mosaico + reproyeccion, con teselas sinteticas
# ----------------------------------------------------------------------
def _escribir_teselas_sinteticas(carpeta, caso, paso=0.001, dh_rim=60.0,
                                 dh_floor=200.0, patron=nombre_tif_fabdem):
    """Reparte un crater de parametros conocidos entre teselas de 1 grado."""
    os.makedirs(carpeta, exist_ok=True)
    _, lista = teselas_necesarias(**caso)
    lon0, lat0 = caso["lon"], caso["lat"]
    r_rim_km = caso["diametro_km"] / 2.0
    rutas = []

    for la, lo in lista:
        n = int(round(1.0 / paso))
        lons = lo + (np.arange(n) + 0.5) * paso
        lats = la + 1.0 - (np.arange(n) + 0.5) * paso      # norte arriba
        LO, LA = np.meshgrid(lons, lats)
        x = (LO - lon0) * 111.320 * np.cos(np.radians(lat0))
        y = (LA - lat0) * 110.574
        rr = np.hypot(x, y)
        z = crater_profile(rr, 300.0, dh_rim, r_rim_km,
                           0.22 * r_rim_km, dh_floor, 0.55 * r_rim_km)

        ruta = os.path.join(carpeta, patron(la, lo))
        with rasterio.open(ruta, "w", driver="GTiff", height=n, width=n,
                           count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(lo, la + 1.0, paso, paso),
                           nodata=-9999.0) as dst:
            dst.write(z.astype("float32"), 1)
        rutas.append(ruta)
    return sorted(rutas)


def test_mosaico_recupera_el_diametro(tmp_path):
    """La prueba que importa: de cuatro teselas sueltas a un diametro en km."""
    carpeta = str(tmp_path / "src")
    _escribir_teselas_sinteticas(carpeta, SERRA)
    salida = str(tmp_path / "serra.tif")

    construir_tesela_fabdem(SERRA["lon"], SERRA["lat"], SERRA["diametro_km"],
                            salida, carpeta=carpeta)

    spec = TileSpec(name="Serra", lon=SERRA["lon"], lat=SERRA["lat"],
                    diameter_km=SERRA["diametro_km"], dataset="FABDEM")
    m = analyze_tile(salida, spec)

    assert m["error_relativo"] < 0.15, f"diametro {m['diameter_km_medido']:.2f} km"
    assert m["r2"] > 0.90
    assert m["coherence"] > 0.80
    assert not m["edge_limited"]


def test_la_salida_esta_en_utm_con_pixeles_cuadrados(tmp_path):
    carpeta = str(tmp_path / "src")
    _escribir_teselas_sinteticas(carpeta, SERRA)
    salida = str(tmp_path / "serra.tif")
    construir_tesela_fabdem(SERRA["lon"], SERRA["lat"], SERRA["diametro_km"],
                            salida, carpeta=carpeta)

    with rasterio.open(salida) as src:
        assert src.crs.to_string() == "EPSG:32723", "UTM 23S, hemisferio sur"
        assert abs(src.transform.a) == pytest.approx(30.0)
        assert abs(src.transform.e) == pytest.approx(30.0)


def test_vichada_tambien_se_mosaica(tmp_path):
    """Seis teselas, hemisferio norte, ventana de 125 km."""
    carpeta = str(tmp_path / "src")
    _escribir_teselas_sinteticas(carpeta, VICHADA, paso=0.002)
    salida = str(tmp_path / "vichada.tif")
    construir_tesela_fabdem(VICHADA["lon"], VICHADA["lat"],
                            VICHADA["diametro_km"], salida, carpeta=carpeta)

    with rasterio.open(salida) as src:
        assert src.crs.to_string() == "EPSG:32619", "UTM 19N"

    spec = TileSpec(name="Vichada", lon=VICHADA["lon"], lat=VICHADA["lat"],
                    diameter_km=VICHADA["diametro_km"], dataset="FABDEM")
    m = analyze_tile(salida, spec)
    assert m["error_relativo"] < 0.20


def test_falta_una_tesela_no_rompe_pero_avisa(tmp_path, capsys):
    carpeta = str(tmp_path / "src")
    rutas = _escribir_teselas_sinteticas(carpeta, SERRA)
    os.remove(rutas[0])

    salida = str(tmp_path / "s.tif")
    construir_tesela_fabdem(SERRA["lon"], SERRA["lat"], SERRA["diametro_km"],
                            salida, carpeta=carpeta)
    texto = capsys.readouterr().out
    assert "falta" in texto.lower()
    assert "AVISO" in texto, "debe advertir de los huecos resultantes"


def test_carpeta_vacia_da_error_util(tmp_path):
    with pytest.raises(RuntimeError, match="FABDEM"):
        construir_tesela_fabdem(SERRA["lon"], SERRA["lat"],
                                SERRA["diametro_km"], str(tmp_path / "x.tif"),
                                carpeta=str(tmp_path / "vacia"))


# ----------------------------------------------------------------------
# Extraccion desde el ZIP
# ----------------------------------------------------------------------
def test_extrae_solo_las_teselas_que_hacen_falta(tmp_path):
    """El ZIP de 10x10 trae ~100 teselas; solo se sacan las 4 necesarias."""
    fuente = str(tmp_path / "todas")
    _escribir_teselas_sinteticas(fuente, SERRA, paso=0.02)
    # anadir teselas que NO hacen falta
    for la, lo in [(-5, -45), (-6, -44)]:
        ruta = os.path.join(fuente, nombre_tif_fabdem(la, lo))
        with rasterio.open(ruta, "w", driver="GTiff", height=10, width=10,
                           count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(lo, la + 1, 0.1, 0.1)) as d:
            d.write(np.zeros((10, 10), "float32"), 1)

    ruta_zip = str(tmp_path / "bloque.zip")
    with zipfile.ZipFile(ruta_zip, "w") as z:
        for f in os.listdir(fuente):
            z.write(os.path.join(fuente, f), arcname=f"FABDEM/{f}")

    sacados = extraer_de_zip_fabdem(ruta_zip, SERRA["lon"], SERRA["lat"],
                                    SERRA["diametro_km"],
                                    carpeta=str(tmp_path / "out"))
    assert len(sacados) == 4
    assert all("S0" in os.path.basename(s) for s in sacados)


def test_zip_con_otro_patron_de_nombres_avisa(tmp_path, capsys):
    ruta_zip = str(tmp_path / "raro.zip")
    with zipfile.ZipFile(ruta_zip, "w") as z:
        z.writestr("OTRO_NOMBRE_S09W047.tif", b"x")
    with pytest.raises(RuntimeError):
        extraer_de_zip_fabdem(ruta_zip, SERRA["lon"], SERRA["lat"],
                              SERRA["diametro_km"], carpeta=str(tmp_path / "o"))
    texto = capsys.readouterr().out
    assert "no estaban en el ZIP" in texto
