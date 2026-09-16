"""
test_campos_potenciales.py — Pruebas de la Fase 3, todas sin red.

Se fabrica una anomalia sintetica de parametros conocidos, se escribe en los
mismos formatos que entregan ICGEM y NOAA, y se comprueba que el modulo la
lea y la recupere. Misma logica que en las fases anteriores: si no recupera
una anomalia que inventamos nosotros, no sirve para una real.

Ejecutar:  pytest -q test_campos_potenciales.py
"""

import numpy as np
import pytest

from campos_potenciales import (Grilla, grados_a_km, leer_csv_generico,
                                leer_gdf_icgem, perfil_radial, resumen_anomalia,
                                separar_regional_residual)

# Estructura del Vichada segun la literatura. Sigue PENDIENTE DE VERIFICAR
# contra Hernandez et al. (2009); ver el Anexo A del anteproyecto.
LON0, LAT0 = -69.25, 4.50
RADIO_KM = 25.0          # anillo externo de 50 km de diametro


# ----------------------------------------------------------------------
# Utilidades: fabricar una anomalia conocida
# ----------------------------------------------------------------------
def _anomalia_sintetica(amplitud=-15.0, radio_km=RADIO_KM, regional=8.0,
                        paso=0.02, medio_ancho=1.5, ruido=0.0, seed=3,
                        regional_km=400.0):
    """Anomalia gaussiana de radio conocido, sobre un regional de onda larga.

    El regional se modela como una gaussiana ancha DESPLAZADA del centro. Una
    rampa lineal no sirve como prueba: al promediar sobre 36 azimuts se
    cancela sola, y la separacion regional/residual pareceria innecesaria.
    """
    lon = np.arange(LON0 - medio_ancho, LON0 + medio_ancho + paso / 2, paso)
    lat = np.arange(LAT0 - medio_ancho, LAT0 + medio_ancho + paso / 2, paso)
    LO, LA = np.meshgrid(lon, lat)
    x, y = grados_a_km(LO - LON0, LA - LAT0, LAT0)
    rr = np.hypot(x, y)

    val = amplitud * np.exp(-(rr ** 2) / (2 * (radio_km / 1.5) ** 2))
    xr, yr = grados_a_km(LO - LON0 + 2.0, LA - LAT0 + 1.0, LAT0)
    val = val + regional * np.exp(-(xr ** 2 + yr ** 2) / (2 * regional_km ** 2))
    if ruido:
        val = val + np.random.default_rng(seed).normal(0, ruido, val.shape)
    return Grilla(lon, lat, val, "mGal", "sintetica")


def _escribir_gdf(path, grilla):
    """Escribe un .gdf con la misma estructura de cabecera que usa ICGEM."""
    with open(path, "w") as fh:
        fh.write("generating_institute   prueba_semillero\n")
        fh.write("functional             gravity_anomaly_bg\n")
        fh.write("unit                   mgal\n")
        fh.write(f"latitude_parallels     {grilla.lat.size}\n")
        fh.write(f"longitude_parallels    {grilla.lon.size}\n")
        fh.write("end_of_head " + "=" * 40 + "\n")
        for i, la in enumerate(grilla.lat):
            for j, lo in enumerate(grilla.lon):
                fh.write(f"{lo:12.6f} {la:12.6f} {grilla.valores[i, j]:14.6f}\n")
    return path


# ----------------------------------------------------------------------
# Lectura de formatos
# ----------------------------------------------------------------------
def test_lee_gdf_de_icgem(tmp_path):
    original = _anomalia_sintetica(paso=0.1, medio_ancho=0.5)
    g = leer_gdf_icgem(_escribir_gdf(str(tmp_path / "a.gdf"), original))
    assert g.valores.shape == original.valores.shape
    assert np.allclose(g.valores, original.valores, atol=1e-4)
    assert g.unidad.lower().startswith("mgal")


def test_gdf_sin_end_of_head_falla_claro(tmp_path):
    p = tmp_path / "malo.gdf"
    p.write_text("-69.0  4.0  1.0\n-69.1  4.0  2.0\n")
    with pytest.raises(ValueError, match="end_of_head"):
        leer_gdf_icgem(str(p))


def test_csv_filtra_por_ventana(tmp_path):
    """Asi se recorta el CSV global de EMAG2 sin cargarlo entero."""
    p = tmp_path / "global.csv"
    filas = []
    for lo in np.arange(-180, 180, 10.0):
        for la in np.arange(-80, 80, 10.0):
            filas.append(f"{lo} {la} {lo + la}")
    p.write_text("\n".join(filas))

    g = leer_csv_generico(str(p), filtro_lon=(-80, -60), filtro_lat=(0, 20),
                          unidad="nT")
    assert g.lon.min() >= -80 and g.lon.max() <= -60
    assert g.lat.min() >= 0 and g.lat.max() <= 20
    assert g.unidad == "nT"


def test_csv_convierte_longitudes_0_360(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("290.0 4.0 1.0\n290.5 4.0 2.0\n290.0 4.5 3.0\n290.5 4.5 4.0\n")
    g = leer_csv_generico(str(p))
    assert g.lon.max() < 0, "290 deg E debe convertirse a -70 deg"


def test_puntos_incompletos_dan_error_explicito(tmp_path):
    """Si el usuario pidio una lista de puntos en vez de una grilla."""
    p = tmp_path / "d.gdf"
    p.write_text("end_of_head ====\n-69.0 4.0 1.0\n-69.1 4.1 2.0\n-69.2 4.3 3.0\n")
    with pytest.raises(ValueError, match="grilla regular"):
        leer_gdf_icgem(str(p))


# ----------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------
def test_conversion_grados_a_km():
    x, y = grados_a_km(1.0, 1.0, 0.0)
    assert x == pytest.approx(111.32, abs=0.01)
    assert y == pytest.approx(110.574, abs=0.01)


def test_un_grado_de_longitud_se_acorta_con_la_latitud():
    x_ec, _ = grados_a_km(1.0, 0.0, 0.0)
    x_45, _ = grados_a_km(1.0, 0.0, 45.0)
    assert x_45 == pytest.approx(x_ec * np.cos(np.radians(45)), rel=1e-6)


def test_resolucion_en_km():
    g = _anomalia_sintetica(paso=0.02, medio_ancho=1.0)
    rx, ry = g.resolucion_km()
    assert ry == pytest.approx(0.02 * 110.574, rel=0.01)
    # A 4.5 N el coseno vale 0.997, asi que un grado de longitud SIGUE siendo
    # mas largo que uno de latitud: se cruzan cerca de 5.5 grados.
    assert rx == pytest.approx(0.02 * 111.320 * np.cos(np.radians(LAT0)), rel=0.01)
    assert rx > ry


# ----------------------------------------------------------------------
# Perfil radial: recuperar una anomalia de radio conocido
# ----------------------------------------------------------------------
def test_perfil_radial_recupera_la_anomalia():
    g = _anomalia_sintetica(amplitud=-15.0, regional=0.0)
    r, perfiles, medio = perfil_radial(g, LON0, LAT0, r_max_km=60.0)

    assert perfiles.shape[0] == 36
    assert medio[0] == pytest.approx(-15.0, abs=0.5), "minimo en el centro"
    assert abs(medio[-1]) < 1.0, "lejos de la estructura debe tender a cero"
    assert np.nanargmin(medio) < 5, "el minimo esta cerca del centro"


def test_perfil_radial_es_simetrico_en_azimut():
    """Una anomalia circular da 36 perfiles casi identicos."""
    g = _anomalia_sintetica(regional=0.0)
    _, perfiles, medio = perfil_radial(g, LON0, LAT0, r_max_km=60.0)
    dispersion = np.nanstd(perfiles, axis=0)
    assert np.nanmax(dispersion) < 0.3


def test_perfil_fuera_de_la_grilla_da_nan():
    g = _anomalia_sintetica(medio_ancho=0.3)
    _, _, medio = perfil_radial(g, LON0, LAT0, r_max_km=200.0)
    assert np.isnan(medio[-1])


# ----------------------------------------------------------------------
# Separacion regional / residual
# ----------------------------------------------------------------------
def test_separacion_quita_la_tendencia_regional():
    g = _anomalia_sintetica(amplitud=-15.0, regional=10.0)
    _, residual = separar_regional_residual(g, longitud_onda_km=150.0)

    _, _, medio_orig = perfil_radial(g, LON0, LAT0, r_max_km=80.0)
    _, _, medio_res = perfil_radial(residual, LON0, LAT0, r_max_km=80.0)

    # el residual debe volver a ~0 lejos de la estructura; el original no
    assert abs(medio_res[-1]) < abs(medio_orig[-1])
    assert medio_res[0] < -5.0, "la anomalia local debe sobrevivir al filtro"


def test_separacion_conserva_la_suma():
    g = _anomalia_sintetica()
    reg, res = separar_regional_residual(g, longitud_onda_km=150.0)
    assert np.allclose(reg.valores + res.valores, g.valores, atol=1e-9)


def test_longitud_de_onda_muy_corta_da_error():
    g = _anomalia_sintetica(paso=0.1)
    with pytest.raises(ValueError, match="demasiado corta"):
        separar_regional_residual(g, longitud_onda_km=5.0)


# ----------------------------------------------------------------------
# Descriptores
# ----------------------------------------------------------------------
def test_resumen_detecta_anomalia_negativa():
    g = _anomalia_sintetica(amplitud=-15.0, regional=0.0)
    r, _, medio = perfil_radial(g, LON0, LAT0, r_max_km=80.0)
    s = resumen_anomalia(r, medio, RADIO_KM)

    assert s["signo"] == "negativa"
    assert s["contraste_centro"] < -10.0
    assert s["amplitud_pico_a_pico"] == pytest.approx(15.0, abs=1.0)


def test_resumen_detecta_anomalia_positiva():
    """El caso que reporta la literatura para el Vichada: anomalia positiva."""
    g = _anomalia_sintetica(amplitud=+12.0, regional=0.0)
    r, _, medio = perfil_radial(g, LON0, LAT0, r_max_km=80.0)
    s = resumen_anomalia(r, medio, RADIO_KM)

    assert s["signo"] == "positiva"
    assert s["contraste_centro"] > 8.0
