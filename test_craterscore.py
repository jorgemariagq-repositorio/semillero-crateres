"""
test_craterscore.py — Pruebas de la Fase 0.

Regla del semillero: nada entra al repositorio sin pasar estas pruebas.
Si el scoring no recupera los parámetros de un cráter que generamos nosotros
mismos, no tiene sentido apuntarlo a Serra da Cangalha.

Los umbrales de estas pruebas son EMPÍRICOS: se fijaron corriendo el módulo
sobre los sintéticos, no a priori. Si alguien cambia el módulo y una prueba
falla, la pregunta correcta es si el cambio mejoró o empeoró la separación,
no si hay que relajar el umbral.

Ejecutar:  pytest -v test_craterscore.py
"""

import numpy as np
import pytest

from craterscore import (
    BCI_DISC_CEILING,
    basin_circularity,
    crater_profile,
    morphometric_metrics,
    synthetic_crater,
)

SEED = 20260906
WIN = (512, 512)


def _crater(**kw):
    base = dict(shape=WIN, r_rim=70.0, dh_rim=40.0, dh_floor=120.0,
                sigma_r=17.0, sigma_f=42.0, noise_std=2.0, seed=SEED)
    base.update(kw)
    return synthetic_crater(**base)


# ----------------------------------------------------------------------
# 1. El modelo tiene el signo correcto  [C1]
# ----------------------------------------------------------------------
def test_perfil_borde_elevado_piso_deprimido():
    """En el centro el terreno baja; en r_rim sube. El borrador lo tenía al revés."""
    r = np.linspace(0, 200, 401)
    z = crater_profile(r, z0=0.0, dh_rim=40.0, r_rim=60.0,
                       sigma_r=15.0, dh_floor=120.0, sigma_f=25.0)
    assert z[0] < -100.0, "el piso debe estar deprimido respecto a z0"
    assert z[np.argmin(np.abs(r - 60.0))] > 25.0, "el borde debe estar elevado"
    assert abs(r[int(np.argmax(z))] - 60.0) < 6.0, "el máximo debe caer en r_rim"


# ----------------------------------------------------------------------
# 2. Recuperación de parámetros conocidos
# ----------------------------------------------------------------------
@pytest.mark.parametrize("r_rim_true", [45.0, 70.0, 110.0])
def test_recupera_radio_del_borde(r_rim_true):
    dem = _crater(r_rim=r_rim_true, sigma_r=0.25 * r_rim_true,
                  sigma_f=0.60 * r_rim_true)
    m = morphometric_metrics(dem)
    err = abs(m["r_rim_px"] - r_rim_true) / r_rim_true
    assert err < 0.15, f"error del {err:.0%} en r_rim ({m['r_rim_px']:.1f})"
    assert m["r2"] > 0.90
    assert m["coherence"] > 0.80
    assert m["ring_contrast"] > 2.0
    assert m["bci"] > 0.70
    assert not m["edge_limited"]


def test_crater_degradado_sigue_siendo_detectable():
    """Caso realista: relieve residual ~25 m, borde de 8 m, ruido de 4 m."""
    dem = _crater(r_rim=80.0, dh_rim=8.0, dh_floor=25.0, sigma_r=25.0,
                  sigma_f=50.0, trend=(0.05, -0.03), noise_std=4.0)
    m = morphometric_metrics(dem)
    assert m["r2"] > 0.70
    assert m["coherence"] > 0.60
    assert m["ring_contrast"] > 2.0
    assert m["bci"] > 0.70
    assert abs(m["r_rim_px"] - 80.0) / 80.0 < 0.25


# ----------------------------------------------------------------------
# 3. Controles negativos
#    Ninguna métrica rechaza sola a todos: cada negativo cae por una razón
#    distinta. Ese es justamente el resultado que hay que reportar.
# ----------------------------------------------------------------------
def test_terreno_plano_con_ruido():
    rng = np.random.default_rng(SEED)
    m = morphometric_metrics(rng.normal(0.0, 5.0, size=WIN))
    assert m["r2"] < 0.30, "ruido blanco no debe ajustar el modelo de cráter"
    assert m["ring_contrast"] < 0.30
    assert m["bci"] < 0.50


def test_rampa_planar():
    m = morphometric_metrics(_crater(dh_rim=0.0, dh_floor=0.0,
                                     trend=(0.4, 0.15), noise_std=3.0))
    assert m["r2"] < 0.30
    assert m["ring_contrast"] < 0.30


def test_anticrater_rechazado_por_el_contraste_anular():
    """Domo central rodeado de anillo deprimido: ring_contrast debe ser < 0."""
    m = morphometric_metrics(-_crater())
    assert m["ring_contrast"] < 0.0
    assert m["bci"] < 0.30


def test_valle_lineal_marcado_como_limitado_por_la_ventana():
    """Un valle gaussiano recto ajusta el modelo con R^2 = 1.00.

    Es el falso positivo más peligroso y NINGUNA de las métricas del borrador
    lo rechaza. Lo rechaza el flag `edge_limited`: el 'borde' ajustado queda
    en el extremo de la ventana, o sea que el terreno nunca vuelve al nivel
    de fondo. De ahí la regla operativa: ventana >= 2.5 x diámetro esperado.
    """
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:WIN[0], 0:WIN[1]].astype(float)
    dem = -80.0 * np.exp(-((xx - WIN[1] / 2) ** 2) / (2 * 40.0 ** 2))
    dem = dem + rng.normal(0.0, 3.0, size=dem.shape)
    m = morphometric_metrics(dem)
    assert m["r2"] > 0.9, "documentamos que SÍ ajusta: ese es el problema"
    assert m["edge_limited"] is True
    assert m["ring_contrast"] < 2.0


def test_elipse_alargada_baja_la_coherencia():
    circ = morphometric_metrics(_crater(ellipticity=1.0))
    elon = morphometric_metrics(_crater(ellipticity=2.2))
    assert elon["coherence"] < 0.6 * circ["coherence"]


# ----------------------------------------------------------------------
# 4. BCI: perímetro y regresión del bug del borrador  [C2] [C3]
# ----------------------------------------------------------------------
def _disco(n=401, radio=150.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2.0
    return np.hypot(xx - c, yy - c) <= radio


def test_bci_disco_define_el_techo():
    """Un disco perfecto da 0.894, no 1.00: la digitalización del contorno
    sobrestima el perímetro ~6%. Cualquier umbral se fija contra ESE techo."""
    assert basin_circularity(_disco(), "contour") == pytest.approx(
        BCI_DISC_CEILING, abs=0.02)


def test_metodo_pixel_da_valor_imposible():
    assert basin_circularity(_disco(), "pixel") > 1.0


def test_regresion_bug_borrador_bci_inflado():
    """La fórmula literal del v2.0 cuenta solo bordes inferior y derecho
    (precedencia & antes que |). Infla el BCI de un disco a ~3.4, con lo cual
    el umbral 0.7 del texto lo pasaba absolutamente cualquier cosa."""
    assert basin_circularity(_disco(), "borrador") > 3.0
    assert basin_circularity(np.ones((301, 301), bool), "borrador") > 3.0


def test_bci_ya_no_es_constante():
    """El código v2.0 pasaba una ventana rectangular: BCI = pi/4 para todo."""
    assert basin_circularity(np.ones((301, 301), bool), "contour") == \
        pytest.approx(np.pi / 4, abs=0.03)
    circ = morphometric_metrics(_crater(r_rim=90.0, ellipticity=1.0))["bci"]
    elon = morphometric_metrics(_crater(r_rim=90.0, ellipticity=3.0))["bci"]
    assert circ > 0.80 and elon < circ
    assert abs(circ - np.pi / 4) > 0.05, "no puede seguir clavado en pi/4"


# ----------------------------------------------------------------------
# 5. Robustez
# ----------------------------------------------------------------------
def test_dem_con_nan_no_rompe():
    dem = _crater()
    dem[100:160, 100:160] = np.nan
    m = morphometric_metrics(dem)
    assert np.isfinite(m["r2"]) and np.isfinite(m["r_rim_px"])
    assert m["r2"] > 0.80


def test_dem_constante_no_rompe():
    m = morphometric_metrics(np.full((256, 256), 1200.0))
    assert m["r2"] == 0.0 and m["coherence"] == 0.0


def test_centro_descentrado():
    dem = _crater(center=(300.0, 220.0), r_rim=60.0)
    m = morphometric_metrics(dem, cx=300.0, cy=220.0, r_max=150.0)
    assert abs(m["r_rim_px"] - 60.0) / 60.0 < 0.15
