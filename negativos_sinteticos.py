"""
negativos_sinteticos.py — Controles negativos geomorfologicos (Semana 2, E2).

Cada funcion genera un DEM sintetico de algo que NO es un crater de impacto
pero que un detector de circularidad podria confundir con uno. Sirven para
medir por que via cae cada tipo de falso positivo.

Todas devuelven una matriz de alturas en metros, sin georreferenciacion.
"""

import numpy as np

__all__ = ["dolina_karstica", "caldera_pico_central", "meandro_abandonado",
           "domo", "NEGATIVOS"]


def _malla(shape):
    """Coordenadas (dx, dy) respecto al centro de la ventana, en pixeles."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    return xx - (nx - 1) / 2, yy - (ny - 1) / 2


def dolina_karstica(shape=(512, 512), radio=40.0, profundidad=60.0,
                    noise_std=3.0, seed=7):
    """Cuenco circular SIN borde elevado: colapso de una cavidad en caliza.

    Es una depresion perfectamente circular, o sea que la circularidad no la
    descarta. Lo que le falta es el borde levantado por la eyeccion.
    """
    dx, dy = _malla(shape)
    rr = np.hypot(dx, dy)
    z = -profundidad * np.exp(-(rr ** 2) / (2 * (radio / 1.5) ** 2))
    return z + np.random.default_rng(seed).normal(0, noise_std, shape)


def caldera_pico_central(shape=(512, 512), r_rim=70.0, dh_rim=35.0,
                         dh_floor=90.0, dh_peak=70.0, sigma_r=15.0,
                         sigma_f=40.0, sigma_p=18.0, noise_std=3.0, seed=11):
    """Caldera volcanica con domo resurgente en el centro.

    ADVERTENCIA: morfologicamente esto es casi lo mismo que un crater complejo
    con pico central. No esperen que las metricas lo rechacen. Esta aqui
    justamente para demostrar que no lo hacen.
    """
    dx, dy = _malla(shape)
    rr = np.hypot(dx, dy)
    z = (dh_rim * np.exp(-((rr - r_rim) ** 2) / (2 * sigma_r ** 2))
         - dh_floor * np.exp(-(rr ** 2) / (2 * sigma_f ** 2))
         + dh_peak * np.exp(-(rr ** 2) / (2 * sigma_p ** 2)))
    return z + np.random.default_rng(seed).normal(0, noise_std, shape)


def meandro_abandonado(shape=(512, 512), r_arco=80.0, ancho=16.0, realce=45.0,
                       prof_interior=25.0, apertura_deg=150.0, orient_deg=30.0,
                       noise_std=3.0, seed=13):
    """Meandro abandonado: arco elevado ABIERTO en un sector, con cuenco dentro.

    `apertura_deg` es cuantos grados del anillo FALTAN. Con 0 el arco es
    cerrado (indistinguible de un borde de crater); con 200 solo queda algo mas
    de un tercio del anillo. Es el caso para el que se diseno la coherencia
    azimutal.
    """
    dx, dy = _malla(shape)
    rr = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dy, dx)) - orient_deg
    th = (th + 180) % 360 - 180
    arco = realce * np.exp(-((rr - r_arco) ** 2) / (2 * ancho ** 2))
    arco = np.where(np.abs(th) <= (360 - apertura_deg) / 2, arco, 0.0)
    cuenco = -prof_interior * np.exp(-(rr ** 2) / (2 * (r_arco * 0.7) ** 2))
    return arco + cuenco + np.random.default_rng(seed).normal(0, noise_std, shape)


def domo(shape=(512, 512), radio=70.0, altura=90.0, noise_std=3.0, seed=17):
    """Domo: elevacion circular. Diapiro salino, intrusion, pliegue braquianticlinal.

    Es lo contrario de un crater: circular en planta, pero convexo.
    """
    dx, dy = _malla(shape)
    rr = np.hypot(dx, dy)
    z = altura * np.exp(-(rr ** 2) / (2 * (radio / 1.4) ** 2))
    return z + np.random.default_rng(seed).normal(0, noise_std, shape)


#: Registro para recorrer todos de una vez.
NEGATIVOS = {
    "dolina karstica": dolina_karstica,
    "caldera + pico central": caldera_pico_central,
    "meandro abandonado": meandro_abandonado,
    "domo": domo,
}
