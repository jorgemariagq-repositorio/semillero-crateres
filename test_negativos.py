"""
test_negativos.py — Pruebas de los controles negativos geomorfologicos.

Cada prueba fija POR QUE VIA cae ese falso positivo. Si alguien cambia el
modulo de metricas y una de estas falla, la pregunta es si el cambio mejoro o
empeoro la separacion, no si hay que relajar el umbral.

Ejecutar:  pytest -q test_negativos.py
"""

import numpy as np
import pytest

from craterscore import morphometric_metrics, synthetic_crater
from negativos_sinteticos import (caldera_pico_central, dolina_karstica, domo,
                                  meandro_abandonado)

WIN = (512, 512)


@pytest.fixture(scope="module")
def crater():
    return morphometric_metrics(synthetic_crater(
        shape=WIN, r_rim=70, dh_rim=40, dh_floor=120,
        sigma_r=17, sigma_f=42, noise_std=3, seed=1))


# ----------------------------------------------------------------------
# Dolina karstica: cae por coherencia y por circularidad
# ----------------------------------------------------------------------
def test_dolina_no_tiene_borde_coherente(crater):
    m = morphometric_metrics(dolina_karstica())
    assert m["r2"] > 0.9, "documentamos que SI ajusta: R2 no la descarta"
    assert m["coherence"] < 0.5 * crater["coherence"]
    assert m["bci"] < 0.30


def test_dolina_radio_ajustado_no_es_fisico():
    """Sin borde real, el modelo pone r_rim donde le conviene, no donde hay algo."""
    m = morphometric_metrics(dolina_karstica(radio=40.0))
    assert m["r_rim_px"] > 100.0, "el radio ajustado no corresponde a la dolina"


def test_dolina_grande_dispara_edge_limited():
    """Al crecer, la dolina llena la ventana. Es la regla de 2.5 diametros."""
    assert morphometric_metrics(dolina_karstica(radio=80.0))["edge_limited"]


# ----------------------------------------------------------------------
# Caldera con pico central: NO se puede rechazar. Resultado negativo,
# y es el mas importante de los cuatro.
# ----------------------------------------------------------------------
def test_caldera_es_indistinguible_de_un_crater(crater):
    """La morfometria por si sola NO separa una caldera resurgente de un
    crater complejo. Esta prueba existe para que ese limite quede escrito y
    nadie lo descubra por sorpresa con datos reales."""
    m = morphometric_metrics(caldera_pico_central())
    assert m["coherence"] > 0.85
    assert m["ring_contrast"] > 5.0
    assert m["bci"] > 0.85
    assert abs(m["coherence"] - crater["coherence"]) < 0.15


def test_caldera_solo_baja_el_r2(crater):
    """Unica senal: el modelo no tiene termino de pico central, asi que ajusta
    algo peor. Es una diferencia pequena y NO sirve como criterio."""
    m = morphometric_metrics(caldera_pico_central())
    assert m["r2"] < crater["r2"]
    assert m["r2"] > 0.80, "pero sigue siendo un ajuste bueno"


# ----------------------------------------------------------------------
# Meandro abandonado: cae por coherencia, y de forma gradual
# ----------------------------------------------------------------------
def test_meandro_abierto_baja_la_coherencia(crater):
    m = morphometric_metrics(meandro_abandonado(apertura_deg=200.0))
    assert m["coherence"] < 0.5
    assert m["coherence"] < 0.6 * crater["coherence"]


def test_coherencia_decrece_con_la_apertura():
    """Monotonia: a mas anillo faltante, menos coherencia."""
    valores = [morphometric_metrics(meandro_abandonado(apertura_deg=a))["coherence"]
               for a in (0.0, 120.0, 200.0)]
    assert valores[0] > valores[1] > valores[2]


def test_meandro_poco_abierto_NO_se_rechaza():
    """Limite honesto del metodo: con 150 grados faltantes la coherencia sigue
    en ~0.79, cerca de un crater real. Un arco moderadamente cerrado pasa."""
    m = morphometric_metrics(meandro_abandonado(apertura_deg=150.0))
    assert m["coherence"] > 0.70


# ----------------------------------------------------------------------
# Domo: cae por el contraste anular, que es la unica metrica con signo
# ----------------------------------------------------------------------
def test_domo_da_contraste_anular_negativo():
    m = morphometric_metrics(domo())
    assert m["ring_contrast"] < 0.0, "el centro esta mas alto que el anillo"
    assert m["bci"] < 0.30


def test_domo_ajusta_bien_pero_no_es_crater():
    """R2 = 0.98: el ajuste es bueno. Sin el signo del contraste anular,
    este falso positivo pasaria."""
    m = morphometric_metrics(domo())
    assert m["r2"] > 0.90
    assert m["ring_contrast"] < 0.0


# ----------------------------------------------------------------------
# Sintesis: ninguna metrica sola rechaza a los cuatro
# ----------------------------------------------------------------------
def test_ninguna_metrica_rechaza_sola_a_todos():
    """El resultado principal de la Fase 0, como prueba automatica."""
    negativos = [dolina_karstica(), caldera_pico_central(),
                 meandro_abandonado(apertura_deg=200.0), domo()]
    ms = [morphometric_metrics(d) for d in negativos]

    assert any(m["r2"] > 0.90 for m in ms), "R2 no rechaza a todos"
    assert any(m["coherence"] > 0.80 for m in ms), "la coherencia tampoco"
    assert any(m["ring_contrast"] > 5.0 for m in ms), "el contraste tampoco"
    assert any(m["bci"] > 0.85 for m in ms), "el BCI tampoco"
