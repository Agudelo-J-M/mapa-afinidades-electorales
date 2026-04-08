# Mapa de Afinidades Electorales — Colombia

Este proyecto genera un grafo de afinidades electorales usando los archivos de entrada:
- `electoral_nodos.csv`
- `electoral_aristas.csv`
- `electoral_README.json`

## Requisitos

Instalar las librerías necesarias:

```bash
pip install pandas networkx pyvis python-louvain
```

## Ejecución

Desde la carpeta del proyecto:

```bash
python main.py
```

## Qué hace el script

- Carga los nodos y aristas de los CSV usando `pandas`.
- Construye un grafo ponderado con `networkx`.
- Detecta comunidades con Louvain (`community_louvain`).
- Genera una visualización interactiva HTML con `pyvis`.
- Responde automáticamente preguntas de análisis y del `electoral_README.json`.
- Calcula métricas clave: modularidad, densidad, número de comunidades y centralidad promedio.
- Identifica nodos puente y simula su eliminación.
- Compara configuraciones de resolución de Louvain si el usuario lo solicita.

## Salida esperada

El script genera un archivo HTML en el directorio del proyecto, por ejemplo:

- `graph_louvain_res_1.0.html`
- `graph_config_1.html`
- `graph_config_2.html`

Abra el archivo HTML en un navegador para explorar el grafo con colores por comunidad y tamaños por centralidad.

## Notas

- El grafo se trata como no dirigido y ponderado para analizar afinidades y comunidades de forma estable.
- Si los datos contienen atributos demográficos adicionales, el script intenta generar subgrafos y compararlos.
