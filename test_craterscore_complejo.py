"""
test_craterscore_complejo.py — Modelo de crater complejo.

Estas pruebas nacen de un fallo real. En la primera corrida sobre dato real
(Serra da Cangalha, GLO-30, semana 3) el modelo simple devolvio un diametro de
2,93 km para una estructura de 13,7 km, con R2 = 0,19 y contraste anular
negativo. No era un error de medicion: era la firma de una morfologia que el
modelo no contemplaba.

El sintetico de referencia de abajo esta calibrado con los numeros medidos en
ese perfil real. Si alguien cambia el modulo y estas pruebas fallan, esta
rompiendo la unica calibracion contra dato real que tiene el proyecto.

Ejecutar:  pytest -q test_craterscore_complejo.py
"""

import numpy as np
import pytest

from craterscore import morphometric_metrics
from craterscore_complejo import (ajustar_perfil_complejo, contraste_foso,
                                  crater_complejo_sintetico,
                                  metricas_complejas, perfil_complejo)

PX = 30.0                    # tamano de pixel, m
R_RIM_PX = 6850.0 / PX       # 13,7 km de diametro (Kenkmann et al., 2011)


def _serra_sintetico(noise_std=6.0, seed=1):
    """Calibrado con el perfil real de Serra da Cangalha sobre GLO-30."""
    return crater_complejo_sintetico(
        shape=(1084, 1084), r_rim=R_RIM_PX, dh_peak=195.0, sigma_p=1200 / PX,
        dh_moat=25.0, r_moat=3500 / PX, sigma_m=1500 / PX,
        dh_rim=12.0, sigma_r=900 / PX, noise_std=noise_std, seed=seed)


# ----------------------------------------------------------------------
# 1. El modelo tiene la forma correcta
# ----------------------------------------------------------------------
def test_perfil_tiene_pico_foso_y_borde():
    r = np.linspace(0, 400, 801)
    z = perfil_complejo(r, 0.0, 195.0, 40.0, 25.0, 117.0, 50.0,
                        12.0, 228.0, 30.0)
    assert z[0] > 150.0, "levantamiento central"
    i_foso = int(np.argmin(z[(r > 60) & (r < 190)])) + int(np.sum(r <= 60))
    assert z[i_foso] < -15.0, "foso anular deprimido"
    banda = (r > 200) & (r < 255)
    assert z[banda].max() > z[i_foso] + 20.0, "borde por encima del foso"


def test_sin_pico_se_reduce_a_crater_simple():
    r = np.linspace(0, 400, 801)
    z = perfil_complejo(r, 0.0, 0.0, 40.0, 100.0, 0.0, 80.0, 40.0, 228.0, 30.0)
    assert z[0] < -80.0, "con r_moat=0 el foso queda en el centro"
    assert np.argmax(z) > 300


# ----------------------------------------------------------------------
# 2. Reproduce el fallo observado y lo corrige
# ----------------------------------------------------------------------
def test_el_modelo_simple_falla_como_en_el_dato_real():
    """Regresion del fallo de la semana 3. Documenta que el modelo simple
    devuelve ~1,5 km de radio para una estructura de 6,85 km."""
    m = morphometric_metrics(_serra_sintetico())
    assert m["r2"] < 0.60, "el ajuste simple es malo, como en el dato real"
    assert m["r_rim_px"] < 0.4 * R_RIM_PX, "se engancha a un anillo interno"
    assert m["ring_contrast"] < 0.0, "contraste centro-borde negativo"


def test_el_modelo_complejo_recupera_el_radio():
    m = metricas_complejas(_serra_sintetico(), r_rim_conocido=R_RIM_PX)
    err = abs(m["r_rim_px"] - R_RIM_PX) / R_RIM_PX
    assert err < 0.10, f"error del {err:.0%} en r_rim"
    assert m["r2"] > 0.90
    assert m["coherence"] > 0.70
    assert not m["edge_limited"]


def test_clasifica_la_morfologia():
    assert metricas_complejas(_serra_sintetico(),
                              r_rim_conocido=R_RIM_PX)["tipo"] == "complejo"


# ----------------------------------------------------------------------
# 3. El contraste foso-borde es positivo en las dos morfologias
# ----------------------------------------------------------------------
def test_realce_del_borde_es_positivo_en_un_complejo():
    m = metricas_complejas(_serra_sintetico(), r_rim_conocido=R_RIM_PX)
    assert m["realce_borde"] > 10.0, "el borde se eleva sobre el foso"
    assert m["ring_contrast"] < 0.0, "pero el centro-borde sigue negativo"


def test_el_foso_queda_entre_el_pico_y_el_borde():
    m = metricas_complejas(_serra_sintetico(), r_rim_conocido=R_RIM_PX)
    assert 0.25 * R_RIM_PX < m["r_foso_px"] < 0.85 * R_RIM_PX


def test_contraste_foso_con_perfil_plano():
    r = np.arange(1.0, 200.0)
    realce, _ = contraste_foso(r, np.zeros_like(r), 100.0)
    assert abs(realce) < 1e-6


# ----------------------------------------------------------------------
# 4. Fijar r_rim evita el enganche al anillo interno
# ----------------------------------------------------------------------
def test_dos_anillos_el_ajuste_libre_puede_equivocarse():
    """Estructura con un anillo interno prominente, como el de 1,6 km que la
    literatura describe en Serra da Cangalha."""
    base = _serra_sintetico(noise_std=4.0)
    ny, nx = base.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    rr = np.hypot(xx - (nx - 1) / 2, yy - (ny - 1) / 2)
    con_anillo = base + 60.0 * np.exp(-((rr - 1600 / PX) ** 2) / (2 * (400 / PX) ** 2))

    fijo = metricas_complejas(con_anillo, r_rim_conocido=R_RIM_PX)
    assert abs(fijo["r_rim_px"] - R_RIM_PX) / R_RIM_PX < 0.15, (
        "con r_rim fijado debe respetar el catalogo")


def test_tolerancia_acota_el_ajuste():
    r = np.arange(1.0, 250.0)
    z = perfil_complejo(r, 0, 195, 40, 25, 117, 50, 12, 228, 30)
    popt, _ = ajustar_perfil_complejo(r, z, r_rim_conocido=228.0,
                                      tolerancia=0.10)
    assert 205.0 <= popt[7] <= 249.0


# ----------------------------------------------------------------------
# 5. Robustez
# ----------------------------------------------------------------------
def test_perfil_plano_no_rompe():
    m = metricas_complejas(np.full((256, 256), 500.0))
    assert m["r2"] == 0.0


def test_dem_con_nan():
    d = _serra_sintetico()
    d[200:300, 200:300] = np.nan
    m = metricas_complejas(d, r_rim_conocido=R_RIM_PX)
    assert np.isfinite(m["r2"]) and m["r2"] > 0.80


def test_crater_simple_tambien_se_ajusta():
    """El modelo complejo debe seguir sirviendo para un cuenco sin pico."""
    from craterscore import synthetic_crater
    d = synthetic_crater(shape=(512, 512), r_rim=70, dh_rim=40, dh_floor=120,
                         sigma_r=17, sigma_f=42, noise_std=2, seed=1)
    m = metricas_complejas(d, r_rim_conocido=70.0)
    assert m["r2"] > 0.90
    assert abs(m["r_rim_px"] - 70.0) / 70.0 < 0.15
    assert m["tipo"] == "simple"
