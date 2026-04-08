import json
import os
import sys
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd
from community import community_louvain
from pyvis.network import Network


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_FILE = os.path.join(DATA_DIR, "electoral_nodos.csv")
EDGE_FILE = os.path.join(DATA_DIR, "electoral_aristas.csv")
METADATA_FILE = os.path.join(DATA_DIR, "electoral_README.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    return pd.read_csv(path, dtype=str).fillna("")


def find_column(df, candidates, required=True):
    normalized = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    if required:
        raise ValueError(f"No se encontró ninguna columna válida entre: {candidates}")
    return None


def infer_node_columns(nodes_df):
    node_id_col = find_column(nodes_df, ["node_id", "id", "identificador", "nodo"])
    type_col = find_column(nodes_df, ["tipo", "type", "category", "categoria", "node_type"], required=False)
    name_col = find_column(nodes_df, ["nombre", "name", "label", "titulo"], required=False)
    return node_id_col, type_col, name_col


def infer_edge_columns(edges_df):
    source_col = find_column(edges_df, ["origen", "source", "src", "from", "u"], required=True)
    target_col = find_column(edges_df, ["destino", "target", "dst", "to", "v"], required=True)
    weight_col = find_column(edges_df, ["peso", "weight", "score", "value", "afinidad", "affinity", "votos_estimados"], required=False)
    edge_type_col = find_column(edges_df, ["tipo_arista", "edge_type", "type", "relation", "relacion"], required=False)
    return source_col, target_col, weight_col, edge_type_col


def safe_numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_graph(nodes_df, edges_df):
    node_id_col, type_col, name_col = infer_node_columns(nodes_df)
    source_col, target_col, weight_col, edge_type_col = infer_edge_columns(edges_df)

    G = nx.Graph()

    # Agregar nodos con atributos.
    for _, row in nodes_df.iterrows():
        node_id = str(row[node_id_col]).strip()
        if not node_id:
            continue
        attrs = {col: value for col, value in row.items() if value != ""}
        # Etiqueta más clara para visualización
        attrs["name"] = attrs.get(name_col, node_id)
        if type_col:
            attrs["type"] = attrs.get(type_col, "desconocido")
        else:
            attrs["type"] = "desconocido"
        G.add_node(node_id, **attrs)

    # Agregar aristas ponderadas.
    for _, row in edges_df.iterrows():
        source = str(row[source_col]).strip()
        target = str(row[target_col]).strip()
        if not source or not target:
            continue
        if source not in G or target not in G:
            continue
        raw_weight = safe_numeric(row[weight_col]) if weight_col else None
        weight = raw_weight if raw_weight is not None else 1.0
        if weight <= 0:
            weight = 1.0
        edge_attrs = {"weight": weight}
        if edge_type_col and row[edge_type_col] != "":
            edge_attrs["edge_type"] = row[edge_type_col]
        for col, value in row.items():
            if col not in {source_col, target_col, weight_col, edge_type_col} and value != "":
                edge_attrs[col] = value
        if G.has_edge(source, target):
            existing = G[source][target]
            existing_weight = existing.get("weight", 1.0)
            existing["weight"] = existing_weight + weight
            # fusionar metadatos si existen
            for key, val in edge_attrs.items():
                if key not in existing:
                    existing[key] = val
        else:
            G.add_edge(source, target, **edge_attrs)

    return G


def summarize_metadata(metadata):
    print("\n=== METADATA DEL DATASET ===")
    print(f"Nombre del dataset: {metadata.get('dataset', 'Desconocido')}")
    print(f"Fecha de elección: {metadata.get('fecha_eleccion', 'No especificada')}")
    print(f"Nota metodológica: {metadata.get('nota_metodologica', 'Sin nota')}\n")
    print("Tipos de nodos:")
    for k, v in metadata.get("tipos_nodos", {}).items():
        print(f" - {k}: {v}")
    print("Tipos de aristas:")
    for k, v in metadata.get("tipos_aristas", {}).items():
        print(f" - {k}: {v}")
    print("Preguntas del dataset:")
    for pregunta in metadata.get("preguntas_que_responde_el_grafo", []):
        print(f" - {pregunta}")
    print()


def compute_graph_metrics(G, partition):
    modularity = community_louvain.modularity(partition, G, weight="weight")
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    community_sizes = {comm: len(nodes) for comm, nodes in communities.items()}
    density = nx.density(G)
    degree_cent = nx.degree_centrality(G)
    avg_centrality = sum(degree_cent.values()) / max(1, len(degree_cent))
    return {
        "modularity": modularity,
        "num_communities": len(communities),
        "community_sizes": community_sizes,
        "density": density,
        "avg_degree_centrality": avg_centrality,
    }


def node_display_name(G, node):
    return G.nodes[node].get("name", node)


def generate_color_map(partition):
    palette = [
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
        "#ff7f00",
        "#6a3d9a",
        "#b15928",
        "#a6cee3",
        "#b2df8a",
        "#fb9a99",
        "#fdbf6f",
        "#cab2d6",
    ]
    color_map = {}
    for i, community in enumerate(sorted(set(partition.values()))):
        color_map[community] = palette[i % len(palette)]
    return color_map


def highlight_bridge_nodes(bridge_centrality, top_n=5):
    sorted_bridges = sorted(bridge_centrality.items(), key=lambda item: item[1], reverse=True)
    return [node for node, _ in sorted_bridges[:top_n]]


def build_interactive_graph(G, partition, filename, show_labels=False, highlight_nodes=None):
    color_map = generate_color_map(partition)
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#000000")
    net.force_atlas_2based()

    for node in G.nodes():
        data = G.nodes[node]
        comm = partition.get(node, -1)
        title = f"{node_display_name(G, node)}<br>Tipo: {data.get('type', 'desconocido')}<br>Comunidad: {comm}" \
                + "<br>".join([f"{k}: {v}" for k, v in data.items() if k not in {"name", "type"}])
        size = 15 + int(degree_cent.get(node, 0) * 80)
        color = color_map.get(comm, "#cccccc")
        border_width = 4 if highlight_nodes and node in highlight_nodes else 1
        net.add_node(
            node,
            label=node_display_name(G, node) if show_labels else "",
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
        )

    for source, target, attrs in G.edges(data=True):
        net.add_edge(source, target, value=attrs.get("weight", 1.0), title=f"Peso: {attrs.get('weight', 1.0)}")

    net.show(filename)
    print(f"Visualización interactiva generada en: {filename}")


def answer_base_questions(G, partition):
    print("\n=== RESPUESTAS AUTOMÁTICAS: PREGUNTAS BASE ===")
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    departments = [n for n, d in G.nodes(data=True) if d.get("type") == "departamento"]
    candidates = [n for n, d in G.nodes(data=True) if d.get("type") == "candidato"]
    media = [n for n, d in G.nodes(data=True) if d.get("type") == "medio"]
    demographic = [n for n, d in G.nodes(data=True) if d.get("type") == "franja_demografica"]

    print("1. ¿Qué departamentos tienen perfiles similares?")
    similar = defaultdict(list)
    for comm, nodes in communities.items():
        deps = [node_display_name(G, n) for n in nodes if n in departments]
        if deps:
            similar[comm] = deps
    for comm, deps in similar.items():
        print(f" - Comunidad {comm}: {', '.join(deps)}")

    print("\n2. ¿Los grupos coinciden con regiones geográficas o ideológicas?")
    if departments:
        region_labels = defaultdict(list)
        for n in departments:
            region = G.nodes[n].get("region", "desconocido")
            region_labels[partition[n]].append(region)
        for comm, regs in region_labels.items():
            top = Counter(regs).most_common(3)
            print(f" - Comunidad {comm}: regiones principales: {', '.join([f'{r} ({c})' for r, c in top])}")
    else:
        print(" - No se detectaron nodos de tipo departamento.")

    print("\n3. ¿Qué medios están asociados a qué candidatos?")
    if media and candidates:
        for m in media:
            neighbors = [n for n in G.neighbors(m) if n in candidates]
            names = [node_display_name(G, n) for n in neighbors]
            print(f" - Medio {node_display_name(G, m)}: candidatos asociados: {', '.join(names) if names else 'ninguno'}")
    else:
        print(" - No hay nodos de tipo medio o candidato suficientes.")

    print("\n4. ¿Qué franjas demográficas son más homogéneas?")
    if demographic:
        for comm, nodes in communities.items():
            fr = [node_display_name(G, n) for n in nodes if n in demographic]
            if fr:
                print(f" - Comunidad {comm}: franjas demográficas: {', '.join(fr)}")
    else:
        print(" - No hay nodos de tipo franja_demografica detectados.")

    print("\n5. ¿Qué tan separados están los grupos?")
    modularity = community_louvain.modularity(partition, G, weight="weight")
    print(f" - Modularidad medida: {modularity:.4f} (más alta indica grupos más separados)")


def answer_readme_questions(G, metadata, partition):
    print("\n=== RESPUESTAS AUTOMÁTICAS: PREGUNTAS DEL README ===")
    for pregunta in metadata.get("preguntas_que_responde_el_grafo", []):
        print(f" - {pregunta}")
    print("\nInterpretación rápida:")
    if G.number_of_nodes() == 0:
        print(" - El grafo está vacío.")
        return
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    print(" - Se identificaron", len(communities), "comunidades.")
    for comm, nodes in communities.items():
        labels = [G.nodes[n].get("type", "?") for n in nodes]
        type_counts = Counter(labels).most_common(4)
        print(f"   * Comunidad {comm}: tipos principales: {type_counts}")


def simulate_bridge_elimination(G, top_n=3):
    betweenness = nx.betweenness_centrality(G, weight="weight")
    bridge_nodes = highlight_bridge_nodes(betweenness, top_n=top_n)
    print("\n=== NODOS PUENTE (BETWEENNESS) ===")
    for node in bridge_nodes:
        print(f" - {node_display_name(G, node)}: betweenness={betweenness[node]:.4f}, tipo={G.nodes[node].get('type', 'desconocido')}" )

    if not bridge_nodes:
        print("No se identificaron nodos puente.")
        return bridge_nodes, None

    H = G.copy()
    H.remove_nodes_from(bridge_nodes)
    print("\nSimulación: eliminación de nodos puente y recalculo de comunidades")
    if H.number_of_nodes() == 0:
        print(" - El grafo queda vacío tras eliminar los nodos puente.")
        return bridge_nodes, None
    partition_H = community_louvain.best_partition(H, weight="weight")
    metrics_H = compute_graph_metrics(H, partition_H)
    print(f" - Modularidad tras eliminación: {metrics_H['modularity']:.4f}")
    print(f" - Número de comunidades tras eliminación: {metrics_H['num_communities']}")
    return bridge_nodes, metrics_H


def detect_demographic_attributes(nodes_df):
    attrs = {}
    for col in nodes_df.columns:
        low = col.lower()
        if "edad" in low or "estrato" in low or "educacion" in low or "rural" in low or "etnia" in low:
            attrs[col] = nodes_df[col].unique().tolist()
    return attrs


def build_demographic_subgraphs(G, nodes_df):
    demo_attrs = detect_demographic_attributes(nodes_df)
    if not demo_attrs:
        print("\nNo se detectaron atributos demográficos compatibles en los nodos.")
        return {}
    subgraphs = {}
    for attr, values in demo_attrs.items():
        if len(values) > 6:
            values = sorted(values)[:6]
        for value in values:
            selected_nodes = [n for n, d in G.nodes(data=True) if d.get(attr) == value]
            if not selected_nodes:
                continue
            H = G.subgraph(selected_nodes).copy()
            if H.number_of_nodes() > 0:
                subgraphs[f"{attr}={value}"] = H
    return subgraphs


def compare_resolution_configs(G, configs):
    print("\n=== COMPARACIÓN DE PARÁMETROS DE LOUVAIN ===")
    results = []
    for label, resolution in configs.items():
        partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
        metrics = compute_graph_metrics(G, partition)
        results.append((label, resolution, metrics, partition))
        print(f" - {label}: resolution={resolution}, modularity={metrics['modularity']:.4f}, comunidades={metrics['num_communities']}")
    return results


def present_comparative_results(results):
    print("\nResumen comparativo:")
    for label, resolution, metrics, _ in results:
        print(f" * {label} (res={resolution}): modularidad={metrics['modularity']:.4f}, comunidades={metrics['num_communities']}, densidad={metrics['density']:.4f}")


def prompt_resolution(default=1.0):
    try:
        value = input(f"Ingrese resolución de Louvain [predeterminado={default}]: ").strip()
        return float(value) if value else default
    except ValueError:
        print("Valor inválido, usando predeterminado.")
        return default


def prompt_node_type_filter(G):
    types = sorted({d.get("type", "desconocido") for _, d in G.nodes(data=True)})
    print("Tipos de nodo disponibles:")
    for i, t in enumerate(types, 1):
        print(f" {i}. {t}")
    choice = input("Filtrar por tipo de nodo (número) o ENTER para usar todos: ").strip()
    if not choice:
        return None
    try:
        selected = types[int(choice) - 1]
        return selected
    except Exception:
        print("Selección inválida, usando todos los tipos.")
        return None


def readme_instructions():
    print("\n=== INSTRUCCIONES RÁPIDAS ===")
    print("1. Asegúrate de tener los archivos: electoral_nodos.csv, electoral_aristas.csv, electoral_README.json")
    print("2. Ejecuta: python main.py")
    print("3. Ingresa resolución de Louvain y filtros cuando se solicite.")
    print("4. El resultado interactivo se genera en archivos HTML en el directorio del proyecto.")


def main():
    try:
        nodes_df = load_csv(NODE_FILE)
        edges_df = load_csv(EDGE_FILE)
        metadata = load_json(METADATA_FILE)
    except Exception as exc:
        print("Error al cargar los archivos de entrada:", exc)
        sys.exit(1)

    summarize_metadata(metadata)
    readme_instructions()

    G = build_graph(nodes_df, edges_df)
    print(f"\nGrafo construido: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")
    print("La red se trata como grafo ponderado sin dirección estricta, ideal para detectar comunidades de afinidad.")

    selected_type = prompt_node_type_filter(G)
    if selected_type:
        nodes_to_keep = [n for n, d in G.nodes(data=True) if d.get("type") == selected_type]
        G = G.subgraph(nodes_to_keep).copy()
        print(f"Filtrando por tipo '{selected_type}', nuevo grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")

    resolution = prompt_resolution(1.0)
    partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
    metrics = compute_graph_metrics(G, partition)
    print("\n=== MÉTRICAS PRINCIPALES ===")
    print(f"Modularidad: {metrics['modularity']:.4f}")
    print(f"Número de comunidades: {metrics['num_communities']}")
    print(f"Densidad del grafo: {metrics['density']:.4f}")
    print(f"Centralidad promedio (grado): {metrics['avg_degree_centrality']:.4f}")
    print("Tamaños de comunidades:")
    for comm, size in metrics['community_sizes'].items():
        print(f" - Comunidad {comm}: {size} nodos")

    answer_base_questions(G, partition)
    answer_readme_questions(G, metadata, partition)

    bridge_nodes, bridge_metrics = simulate_bridge_elimination(G, top_n=3)

    subgraphs = build_demographic_subgraphs(G, nodes_df)
    if subgraphs:
        print("\n=== SUBGRAFOS DEMOGRÁFICOS ===")
        for label, H in subgraphs.items():
            print(f" - {label}: {H.number_of_nodes()} nodos, {H.number_of_edges()} aristas")
            if H.number_of_nodes() > 2:
                partition_H = community_louvain.best_partition(H, weight="weight")
                metrics_H = compute_graph_metrics(H, partition_H)
                print(f"   modularidad={metrics_H['modularity']:.4f}, comunidades={metrics_H['num_communities']}")

    compare = input("¿Desea comparar dos configuraciones de resolución? (s/n): ").strip().lower()
    if compare == "s":
        alt_resolution = prompt_resolution(1.5)
        configs = {"config_1": resolution, "config_2": alt_resolution}
        results = compare_resolution_configs(G, configs)
        present_comparative_results(results)
        for label, resolution_value, _, partition_value in results:
            filename = os.path.join(DATA_DIR, f"graph_{label}.html")
            build_interactive_graph(G, partition_value, filename, show_labels=False, highlight_nodes=bridge_nodes)

    output_file = os.path.join(DATA_DIR, f"graph_louvain_res_{resolution}.html")
    build_interactive_graph(G, partition, output_file, show_labels=False, highlight_nodes=bridge_nodes)
    print("\nFinalizado. Abra el HTML generado en un navegador para explorar el grafo interactivamente.")


if __name__ == "__main__":
    main()
