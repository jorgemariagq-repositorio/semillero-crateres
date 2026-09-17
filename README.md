# Detectabilidad morfométrica de estructuras de impacto

Semillero de Astronomía — Pontificia Universidad Javeriana Cali.  
Departamento de Ciencias Naturales y Matemáticas y Departamento de
Ingeniería Electrónica, Facultad de Ingeniería y Ciencias.

Este repositorio contiene el código de un proyecto que **mide hasta dónde
llega** un método morfométrico sobre modelos digitales de elevación abiertos.
No busca descubrir cráteres: el número esperado de estructuras de impacto no
descubiertas y detectables en Colombia es del orden de 0,7, de modo que un
catálogo de candidatos tendría una precisión cercana a cero por construcción.

## Estado

| Objetivo | D (km) | Morfología | Resultado |
|---|---|---|---|
| Meteor Crater (Arizona) | 1,186 | simple | R² 0,994 · coherencia 0,79 · error de diámetro +8,8 % |
| Serra da Cangalha (Brasil) | 13,7 | compleja | R² 0,976 · coherencia 0,38 · realce de borde +12,9 m |
| Uhackatik (Quebec) | 25,0 | compleja | pendiente |
| Estructura del Vichada (Colombia) | 50,0 | — | sin expresión topográfica coherente |
| 7 estructuras confirmadas de Brasil | 3,6–40 | mixta | pendiente |

El modelo de cráter simple falla sobre Serra da Cangalha: converge al anillo
interno de 1,6 km y devuelve un diámetro cinco veces menor que el de catálogo.
Ese fallo motivó `craterscore_complejo.py`. Un banco de pruebas sintético no
podía detectarlo, porque el generador y el estimador compartían el mismo
supuesto geométrico.

## Advertencia sobre la comparación con la distribución nula

El estadístico de prueba es la **máxima coherencia azimutal sobre un barrido
de ocho escalas**, y está sesgado al alza: el máximo de ocho extracciones cae
por aritmética cerca del percentil 89 de la distribución. No tiene
interpretación absoluta.

Su distribución nula debe construirse con **el mismo barrido, el mismo modelo
de elevación y la misma resolución** que el objetivo. Una corrida preliminar
midió el objetivo sobre FABDEM y los controles sobre GLO-30; esa comparación
no es válida y no debe reportarse.

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
!git clone -q https://github.com/jorgemariagq-repositorio/semillero-crateres.git
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

## Estructura

```
.                      modulos y pruebas
docs/                  anteproyecto, guias de procedimiento, articulo, seminario
notebooks/             cuadernos de Colab
```

## Datos

**No se versionan.** Las teselas se descargan con `dem_directo.py` y se guardan
fuera del repositorio. FABDEM está bajo CC BY-NC-SA 4.0 y no se redistribuye.

## Licencia

Código bajo MIT (ver `LICENSE`). Los datos conservan sus propias condiciones.
