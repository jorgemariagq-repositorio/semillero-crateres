"""
dem_directo.py — Teselas de DEM sin Google Earth Engine.

Ruta alternativa a `gee_tiles.py`: descarga desde los repositorios de origen,
sin cuenta de Google, sin cuota y sin tramite institucional.

  FABDEM  Universidad de Bristol (data.bris). Licencia CC BY-NC-SA 4.0.
          Se distribuye como ZIP de 10x10 grados con teselas COG de 1 grado.
  GLO-30  Registro de datos abiertos de AWS, bucket publico anonimo, COG.

La interfaz de salida es IDENTICA a la de `gee_tiles.fetch_tile`: un GeoTIFF
en UTM local, con pixeles cuadrados en metros, listo para `analyze_tile`. Se
puede cambiar de ruta sin tocar nada aguas abajo.

POR QUE HAY QUE MOSAICAR
Las teselas de origen son de 1 grado. Ninguna de las dos ventanas del proyecto
cabe en una sola:
  Serra da Cangalha (13 km)  -> 4 teselas: S09W048 S09W047 S08W048 S08W047
  Estructura del Vichada (50 km) -> 6 teselas: N03W070 N03W069 N04W070
                                    N04W069 N05W070 N05W069
Por eso el modulo calcula que teselas hacen falta, las une, reproyecta a UTM y
recorta. Hacerlo a mano es una fuente de errores silenciosos.

ADVERTENCIA SOBRE LO QUE NO ESTA PROBADO
Toda la logica local (calculo de teselas, mosaico, reproyeccion, recorte,
metricas) esta cubierta por pruebas con archivos sinteticos. Las descargas
reales NO se pudieron probar al escribir el modulo: hay que verificar los
patrones de URL contra el readme de cada repositorio la primera vez. Si una
URL falla, el problema esta casi seguro en el patron de nombres, no en la
logica.
"""

from __future__ import annotations

import math
import os
import zipfile

import numpy as np

__all__ = [
    "URL_GLO30_BASE",
    "URL_FABDEM_DATASET",
    "nombre_tesela",
    "teselas_necesarias",
    "nombre_zip_fabdem",
    "url_glo30",
    "mosaicar_y_reproyectar",
    "construir_tesela_glo30",
    "construir_tesela_fabdem",
    "extraer_de_zip_fabdem",
]

#: Bucket publico de AWS. Acceso anonimo por HTTPS, sin credenciales.
URL_GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

#: Pagina del conjunto de datos en el repositorio de Bristol.
URL_FABDEM_DATASET = ("https://data.bris.ac.uk/data/dataset/"
                      "s5hqmjcdj8yo2ibzi9b4ew3sn")


# ----------------------------------------------------------------------
# Nomenclatura de teselas
# ----------------------------------------------------------------------
def nombre_tesela(lat_int: int, lon_int: int) -> str:
    """Nombre de una tesela de 1 grado por su esquina suroeste.

    >>> nombre_tesela(-9, -47)
    'S09W047'
    >>> nombre_tesela(4, -70)
    'N04W070'
    """
    ns = "N" if lat_int >= 0 else "S"
    ew = "E" if lon_int >= 0 else "W"
    return f"{ns}{abs(lat_int):02d}{ew}{abs(lon_int):03d}"


def teselas_necesarias(lon: float, lat: float, diametro_km: float,
                       factor: float = 2.5):
    """Teselas de 1 grado que cubren la ventana de analisis.

    Devuelve (limites, lista) con limites = (lon_min, lon_max, lat_min,
    lat_max) y lista = [(lat_int, lon_int), ...].

    `factor` es la regla operativa de la Fase 0: la ventana debe medir al menos
    2,5 veces el diametro esperado, o `edge_limited` se dispara.
    """
    if diametro_km <= 0 or factor <= 0:
        raise ValueError("diametro_km y factor deben ser positivos")
    medio_km = diametro_km * factor / 2.0
    dlat = medio_km / 110.574
    dlon = medio_km / (111.320 * math.cos(math.radians(lat)))

    limites = (lon - dlon, lon + dlon, lat - dlat, lat + dlat)
    lista = [(la, lo)
             for la in range(math.floor(limites[2]), math.floor(limites[3]) + 1)
             for lo in range(math.floor(limites[0]), math.floor(limites[1]) + 1)]
    return limites, lista


def nombre_zip_fabdem(lat_int: int, lon_int: int) -> str:
    """ZIP de 10x10 grados de FABDEM que contiene esa tesela.

    Bristol agrupa las teselas en bloques de 10 grados nombrados por sus
    esquinas suroeste y noreste.

    >>> nombre_zip_fabdem(-9, -47)
    'S10W050-N00W040_FABDEM_V1-2.zip'
    >>> nombre_zip_fabdem(4, -70)
    'N00W070-N10W060_FABDEM_V1-2.zip'
    """
    la0 = math.floor(lat_int / 10.0) * 10
    lo0 = math.floor(lon_int / 10.0) * 10
    return (f"{nombre_tesela(la0, lo0)}-"
            f"{nombre_tesela(la0 + 10, lo0 + 10)}_FABDEM_V1-2.zip")


def nombre_tif_fabdem(lat_int: int, lon_int: int) -> str:
    """Nombre del GeoTIFF dentro del ZIP."""
    return f"{nombre_tesela(lat_int, lon_int)}_FABDEM_V1-2.tif"


def url_glo30(lat_int: int, lon_int: int) -> str:
    """URL del COG de GLO-30 en el bucket publico de AWS.

    VERIFICAR la primera vez contra el readme del bucket:
    https://copernicus-dem-30m.s3.amazonaws.com/readme.html
    Si el patron cambio, es una linea de este modulo.
    """
    ns = "N" if lat_int >= 0 else "S"
    ew = "E" if lon_int >= 0 else "W"
    clave = (f"Copernicus_DSM_COG_10_{ns}{abs(lat_int):02d}_00_"
             f"{ew}{abs(lon_int):03d}_00_DEM")
    return f"{URL_GLO30_BASE}/{clave}/{clave}.tif"


# ----------------------------------------------------------------------
# Mosaico + reproyeccion + recorte
# ----------------------------------------------------------------------
def mosaicar_y_reproyectar(fuentes, lon, lat, diametro_km, salida,
                           pixel_m=30.0, factor=2.5, resampleo="bilinear"):
    """Une varias teselas geograficas, reproyecta a UTM local y recorta.

    `fuentes` es una lista de rutas locales o de URLs con prefijo /vsicurl/.
    Las URLs de COG se leen por ventana sobre HTTP, sin bajar el archivo
    entero.

    El resultado es un GeoTIFF con pixeles cuadrados en metros, el mismo
    producto que entrega `gee_tiles.fetch_tile`.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject

    from gee_tiles import tile_bounds_utm, utm_epsg, window_size_px

    if not fuentes:
        raise ValueError("no se paso ninguna fuente")

    limites, _ = teselas_necesarias(lon, lat, diametro_km, factor)
    lon0, lon1, lat0, lat1 = limites
    margen = 0.05                       # colchon para el remuestreo del borde

    abiertos = [rasterio.open(f) for f in fuentes]
    try:
        mosaico, tr_mosaico = merge(
            abiertos,
            bounds=(lon0 - margen, lat0 - margen, lon1 + margen, lat1 + margen))
        perfil = abiertos[0].profile
        nodata = abiertos[0].nodata
        crs_origen = abiertos[0].crs
    finally:
        for a in abiertos:
            a.close()

    n_px = window_size_px(diametro_km, pixel_m, factor)
    crs_utm, bounds_utm, _ = tile_bounds_utm(lon, lat, n_px * pixel_m / 2.0)

    destino = np.full((n_px, n_px), np.nan, dtype="float32")
    tr_destino = rasterio.transform.from_origin(
        bounds_utm[0], bounds_utm[3], pixel_m, pixel_m)

    reproject(
        source=mosaico[0].astype("float32"),
        destination=destino,
        src_transform=tr_mosaico,
        src_crs=crs_origen,
        src_nodata=nodata,
        dst_transform=tr_destino,
        dst_crs=crs_utm,
        dst_nodata=np.nan,
        resampling=getattr(Resampling, resampleo),
    )

    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    perfil.update(driver="GTiff", height=n_px, width=n_px, count=1,
                  dtype="float32", crs=crs_utm, transform=tr_destino,
                  nodata=np.nan, compress="deflate")
    with rasterio.open(salida, "w", **perfil) as dst:
        dst.write(destino, 1)

    nan_frac = float(np.mean(~np.isfinite(destino)))
    print(f"[ok] {salida}  {n_px}x{n_px} px @ {pixel_m:.0f} m  {crs_utm}  "
          f"NaN {nan_frac:.1%}")
    if nan_frac > 0.05:
        print("[AVISO] hay huecos: puede faltar alguna tesela de origen")
    return salida


# ----------------------------------------------------------------------
# GLO-30: lectura directa por HTTP, sin descargar el archivo entero
# ----------------------------------------------------------------------
def construir_tesela_glo30(lon, lat, diametro_km, salida, pixel_m=30.0,
                           factor=2.5, verificar=True):
    """Construye la tesela de GLO-30 leyendo los COG por /vsicurl/.

    No descarga las teselas completas: GDAL pide solo los bloques que caen
    dentro de la ventana. Requiere rasterio compilado con soporte de curl,
    que es lo normal en Colab.
    """
    import rasterio

    _, lista = teselas_necesarias(lon, lat, diametro_km, factor)
    fuentes, faltantes = [], []
    for la, lo in lista:
        url = url_glo30(la, lo)
        ruta = f"/vsicurl/{url}"
        if verificar:
            try:
                with rasterio.open(ruta):
                    pass
            except Exception:
                faltantes.append(nombre_tesela(la, lo))
                continue
        fuentes.append(ruta)

    if faltantes:
        print(f"[AVISO] no se pudieron abrir: {faltantes}")
        print("  Puede ser un hueco real del producto publico, o que el "
              "patron de URL haya cambiado. Verifica en "
              f"{URL_GLO30_BASE}/readme.html")
    if not fuentes:
        raise RuntimeError("ninguna tesela de GLO-30 disponible")

    return mosaicar_y_reproyectar(fuentes, lon, lat, diametro_km, salida,
                                  pixel_m, factor)


# ----------------------------------------------------------------------
# FABDEM: se baja el ZIP y se extraen las teselas que hacen falta
# ----------------------------------------------------------------------
def extraer_de_zip_fabdem(ruta_zip, lon, lat, diametro_km, carpeta="fabdem",
                          factor=2.5):
    """Saca del ZIP solo las teselas de 1 grado que cubren la ventana."""
    _, lista = teselas_necesarias(lon, lat, diametro_km, factor)
    quiero = {nombre_tif_fabdem(la, lo) for la, lo in lista}
    os.makedirs(carpeta, exist_ok=True)

    sacados, dentro = [], set()
    with zipfile.ZipFile(ruta_zip) as z:
        for miembro in z.namelist():
            base = os.path.basename(miembro)
            dentro.add(base)
            if base in quiero:
                destino = os.path.join(carpeta, base)
                if not os.path.exists(destino):
                    with z.open(miembro) as src, open(destino, "wb") as dst:
                        dst.write(src.read())
                sacados.append(destino)

    faltan = quiero - {os.path.basename(s) for s in sacados}
    if faltan:
        print(f"[AVISO] no estaban en el ZIP: {sorted(faltan)}")
        ejemplos = sorted(n for n in dentro if n.endswith('.tif'))[:3]
        if ejemplos:
            print(f"  Nombres reales dentro del ZIP, por ejemplo: {ejemplos}")
            print("  Si el patron difiere, corrige `nombre_tif_fabdem`.")
    if not sacados:
        raise RuntimeError("no se extrajo ninguna tesela del ZIP")
    return sorted(sacados)


def construir_tesela_fabdem(lon, lat, diametro_km, salida, carpeta="fabdem",
                            pixel_m=30.0, factor=2.5):
    """Construye la tesela de FABDEM a partir de los .tif ya extraidos."""
    _, lista = teselas_necesarias(lon, lat, diametro_km, factor)
    fuentes = []
    for la, lo in lista:
        ruta = os.path.join(carpeta, nombre_tif_fabdem(la, lo))
        if os.path.exists(ruta):
            fuentes.append(ruta)
        else:
            print(f"[AVISO] falta {ruta}")
    if not fuentes:
        raise RuntimeError(
            f"no hay teselas de FABDEM en {carpeta!r}. Descarga "
            f"{nombre_zip_fabdem(*lista[0])} de {URL_FABDEM_DATASET} "
            f"y extrae con extraer_de_zip_fabdem().")
    return mosaicar_y_reproyectar(fuentes, lon, lat, diametro_km, salida,
                                  pixel_m, factor)


# ----------------------------------------------------------------------
# Resumen imprimible de lo que hay que descargar
# ----------------------------------------------------------------------
def plan_de_descarga(nombre, lon, lat, diametro_km, factor=2.5):
    """Imprime exactamente que hace falta bajar para un objetivo."""
    limites, lista = teselas_necesarias(lon, lat, diametro_km, factor)
    lon0, lon1, lat0, lat1 = limites
    zips = sorted({nombre_zip_fabdem(la, lo) for la, lo in lista})

    print(f"=== {nombre} ===")
    print(f"Centro          : {lon:.4f}, {lat:.4f}")
    print(f"Diametro        : {diametro_km} km  ->  ventana {factor}x = "
          f"{diametro_km * factor:.1f} km")
    print(f"Ventana (grados): lon [{lon0:.3f}, {lon1:.3f}]  "
          f"lat [{lat0:.3f}, {lat1:.3f}]")
    print(f"Teselas 1 grado : {len(lista)}  ->  "
          f"{', '.join(nombre_tesela(la, lo) for la, lo in lista)}")
    print(f"\nFABDEM: baja {len(zips)} archivo(s) de {URL_FABDEM_DATASET}")
    for z in zips:
        print(f"   {z}")
    print("\nGLO-30: lectura directa por HTTP, sin descarga previa")
    for la, lo in lista:
        print(f"   {url_glo30(la, lo)}")
    return limites, lista
