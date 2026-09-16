# Detectabilidad morfométrica de estructuras de impacto

Semillero de Astronomía y Ciencias Planetarias — Universidad Icesi, Cali.

Este repositorio contiene el código de un proyecto que **mide hasta dónde
llega** un método morfométrico sobre modelos digitales de elevación abiertos.
No busca descubrir cráteres: el número esperado de estructuras de impacto no
descubiertas y detectables en Colombia es del orden de 0,7, de modo que un
catálogo de candidatos tendría una precisión cercana a cero por construcción.

## Estado

| Objetivo | D (km) | Morfología | Resultado |
|---|---|---|---|
| Meteor Crater (Arizona) | 1,186 | simple | R² 0,994 · coherencia 0,79 · error +8,8 % |
| Serra da Cangalha (Brasil) | 13,7 | compleja | en curso |
| Estructura del Vichada (Colombia) | 50 | ? | pendiente |

## Módulos

| Archivo | Función |
|---|---|
| `craterscore.py` | Métricas morfométricas. Modelo de cráter **simple** |
| `craterscore_complejo.py` | Modelo de cráter **complejo** (levantamiento central, foso, borde) |
| `dem_directo.py` | Descarga de teselas sin Google Earth Engine |
| `gee_tiles.py` | Utilidades geoespaciales y extracción vía GEE |
| `campos_potenciales.py` | Anomalías de gravedad y magnetismo (Fase 3) |
| `negativos_sinteticos.py` | Controles negativos geomorfológicos |

## Uso desde Google Colab

```python
!git clone -q https://github.com/USUARIO/semillero-crateres.git
import sys; sys.path.insert(0, "/content/semillero-crateres")
```

## Pruebas

```bash
pip install -r requirements.txt
pytest -q
```

Nada entra al repositorio sin pasar las pruebas.

## Reglas de trabajo

1. Ningún código entra sin haberse ejecutado.
2. Toda función de métrica lleva una prueba con una estructura sintética de
   parámetros conocidos.
3. Las versiones de las librerías quedan fijadas en `requirements.txt`.
4. Toda cifra del informe debe poder regenerarse ejecutando un script.
5. Ninguna referencia se cita sin verificar autor, año y medio de publicación.

## Datos

**No se versionan.** Las teselas se descargan con `dem_directo.py` y se guardan
fuera del repositorio. FABDEM está bajo CC BY-NC-SA 4.0 y no se redistribuye.

## Licencia

Código bajo MIT (ver `LICENSE`). Los datos conservan sus propias condiciones.
