import os

import networkx as nx

from matplotlib import pyplot as plt
from networkx.algorithms.cuts import conductance
from sklearn.cluster import SpectralClustering

from networkx.algorithms.community.quality import modularity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "facebook_combined.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_graph():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find file: {DATA_PATH}")

    return nx.read_edgelist(DATA_PATH, nodetype=int, create_using=nx.Graph())


def create_output_folder():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def display_dataset_description(graph):
    print("\nDATASET DESCRIPTION")
    print("===================")
    print("Dataset: SNAP Facebook combined ego-network")
    print("Nodes: anonymized Facebook users")
    print("Edges: friendship connections between users")
    print("Graph type: undirected")

    print("\nBASIC GRAPH INFORMATION")
    print("=======================")
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Density:", nx.density(graph))
    print("Is connected:", nx.is_connected(graph))


def analyze_degrees(graph):
    degrees = dict(nx.degree(graph))
    degree_values = list(degrees.values())

    average_degree = sum(degree_values) / len(degree_values)
    min_degree = min(degree_values)
    max_degree = max(degree_values)

    print("\nDEGREE ANALYSIS")
    print("===============")
    print(f"Average degree: {average_degree:.2f}")
    print(f"Minimum degree: {min_degree}")
    print(f"Maximum degree: {max_degree}")

    print_top_metric_nodes(degrees, "degree", top_n=10, precision=0)


def analyze_clustering(graph):
    average_clustering = nx.average_clustering(graph)

    clustering_values = nx.clustering(graph)

    print("\nCLUSTERING ANALYSIS")
    print("===================")
    print(f"Average clustering coefficient: {average_clustering:.4f}")

    print_top_metric_nodes(
        clustering_values, "clustering coefficient", top_n=10, precision=4
    )


def analyze_connected_components(graph):
    components = list(nx.connected_components(graph))
    component_sizes = [len(component) for component in components]

    largest_component_size = max(component_sizes)

    print("\nCONNECTED COMPONENTS")
    print("====================")
    print(f"Number of connected components: {len(components)}")
    print(f"Largest connected component size: {largest_component_size}")

    print("\nComponent sizes:")
    for i, size in enumerate(sorted(component_sizes, reverse=True), start=1):
        print(f"Component {i}: {size} nodes")


def print_top_metric_nodes(metric_dict, metric_name, top_n=10, precision=2):
    top_nodes = sorted(metric_dict.items(), key=lambda item: item[1], reverse=True)[
        :top_n
    ]

    print(f"\nTop {top_n} nodes by {metric_name}:")
    for node, score in top_nodes:
        print(f"Node {node}: {score:.{precision}f}")


def analyze_centralities(graph):
    print("\nCENTRALITY ANALYSIS")
    print("===================")

    print("\nDegree centrality")
    degree_centrality = nx.degree_centrality(graph)
    print_top_metric_nodes(degree_centrality, "degree centrality", precision=6)

    print("\nBetweenness centrality")
    betweenness_centrality = nx.betweenness_centrality(graph)
    print_top_metric_nodes(
        betweenness_centrality, "betweenness centrality", precision=6
    )

    print("\nEigenvector centrality")
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
    print_top_metric_nodes(
        eigenvector_centrality, "eigenvector centrality", precision=6
    )

    print("\nCloseness centrality")
    closeness_centrality = nx.closeness_centrality(graph)
    print_top_metric_nodes(closeness_centrality, "closeness centrality", precision=6)


def analyze_graph(graph):
    display_dataset_description(graph)
    analyze_degrees(graph)
    analyze_clustering(graph)
    analyze_connected_components(graph)
    analyze_centralities(graph)


def run_louvain(graph):
    communities = nx.community.louvain_communities(graph, seed=42)
    number_of_communities = len(communities)
    community_sizes = sorted(
        [len(community) for community in communities], reverse=True
    )
    smallest_community_size = min(community_sizes)
    largest_community_size = max(community_sizes)
    avg_community_size = sum(community_sizes) / len(community_sizes)
    network_louvain_modularity = modularity(graph, communities)
    louvain_conductance = calculate_average_conductance(graph, communities)

    print("\nLOUVAIN ANALYSIS")
    print("==================")
    print(f"Number of communities: {number_of_communities}")
    print(f"Community sizes: {community_sizes}")
    print(f"Smallest community size: {smallest_community_size}")
    print(f"Largest community size: {largest_community_size}")
    print(f"Average community size: {avg_community_size:.2f}")
    print(f"Modularity: {network_louvain_modularity:.4f}")
    print(f"Average conductance: {louvain_conductance:.4f}")

    return {
        "communities": communities,
        "number_of_communities": number_of_communities,
        "community_sizes": community_sizes,
        "smallest_community_size": smallest_community_size,
        "largest_community_size": largest_community_size,
        "avg_community_size": avg_community_size,
        "modularity": network_louvain_modularity,
        "conductance": louvain_conductance,
    }


def get_top_degree_subgraph(graph, n=500):
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:n]
    return graph.subgraph(top_nodes).copy()


def get_largest_connected_component_subgraph(graph):
    largest_component_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component_nodes).copy()


def display_subgraph_information(subgraph):
    print("\nSUBGRAPH INFORMATION")
    print("====================")
    print("Subgraph nodes:", subgraph.number_of_nodes())
    print("Subgraph edges:", subgraph.number_of_edges())
    print("Subgraph connected:", nx.is_connected(subgraph))


def run_girvan_newman(graph, k=2):
    print("\nGIRVAN-NEWMAN ANALYSIS")
    print("==================")
    comp = nx.community.girvan_newman(graph)
    final_communities = None
    for communities in comp:
        if len(communities) == k:
            final_communities = tuple(sorted(community) for community in communities)
            break

    if final_communities is None:
        print(f"No partition with exactly {k} communities found.")
        return None

    community_sizes = [len(community) for community in final_communities]
    number_of_communities = len(final_communities)
    smallest_community_size = min(community_sizes)
    largest_community_size = max(community_sizes)
    avg_community_size = sum(community_sizes) / len(community_sizes)
    girvan_newman_modularity = modularity(graph, final_communities)
    girvan_newman_conductance = calculate_average_conductance(graph, final_communities)

    print(f"Number of communities: {number_of_communities}")
    print(f"Community sizes: {community_sizes}")
    print(f"Smallest community size: {smallest_community_size}")
    print(f"Largest community size: {largest_community_size}")
    print(f"Average community size: {avg_community_size:.2f}")
    print(f"Modularity: {girvan_newman_modularity:.4f}")
    print(f"Average conductance: {girvan_newman_conductance:.4f}")

    return {
        "communities": final_communities,
        "number_of_communities": number_of_communities,
        "community_sizes": community_sizes,
        "smallest_community_size": smallest_community_size,
        "largest_community_size": largest_community_size,
        "avg_community_size": avg_community_size,
        "modularity": girvan_newman_modularity,
        "conductance": girvan_newman_conductance,
    }


def run_spectral_clustering(graph, k):
    nodes = list(graph.nodes())
    spectral_clustering_adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)
    spectral_model = SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=42
    )
    labels = spectral_model.fit_predict(spectral_clustering_adjacency_matrix)

    communities = []
    for cluster_id in range(k):
        community = {node for i, node in enumerate(nodes) if labels[i] == cluster_id}
        communities.append(community)

    community_sizes = sorted(
        [len(community) for community in communities], reverse=True
    )

    number_of_communities = len(communities)
    smallest_community_size = min(community_sizes)
    largest_community_size = max(community_sizes)
    avg_community_size = sum(community_sizes) / len(community_sizes)
    spectral_modularity = modularity(graph, communities)
    spectral_conductance = calculate_average_conductance(graph, communities)

    print("\nSPECTRAL CLUSTERING ANALYSIS")
    print("============================")
    print(f"Number of communities: {number_of_communities}")
    print(f"Community sizes: {community_sizes}")
    print(f"Smallest community size: {smallest_community_size}")
    print(f"Largest community size: {largest_community_size}")
    print(f"Average community size: {avg_community_size:.2f}")
    print(f"Modularity: {spectral_modularity:.4f}")
    print(f"Average conductance: {spectral_conductance:.4f}")

    return {
        "communities": communities,
        "number_of_communities": number_of_communities,
        "community_sizes": community_sizes,
        "smallest_community_size": smallest_community_size,
        "largest_community_size": largest_community_size,
        "avg_community_size": avg_community_size,
        "modularity": spectral_modularity,
        "conductance": spectral_conductance,
    }


def calculate_average_conductance(graph, communities):
    conductance_scores = [conductance(graph, community) for community in communities]
    return sum(conductance_scores) / len(conductance_scores)


def display_community_detection_comparison(
    louvain_results,
    spectral_clustering_results,
    girvan_newman_results,
    spectral_small_results,
):
    print("\nCOMMUNITY DETECTION COMPARISON")
    print("==============================")

    print("\nFULL GRAPH COMPARISON")
    print("=====================")
    print(
        f"{'Method':<20}"
        f"{'Communities':<15}"
        f"{'Modularity':<15}"
        f"{'Avg Conductance':<20}"
    )

    print(
        f"{'Louvain':<20}"
        f"{louvain_results['number_of_communities']:<15}"
        f"{louvain_results['modularity']:<15.4f}"
        f"{louvain_results['conductance']:<20.4f}"
    )

    print(
        f"{'Spectral':<20}"
        f"{spectral_clustering_results['number_of_communities']:<15}"
        f"{spectral_clustering_results['modularity']:<15.4f}"
        f"{spectral_clustering_results['conductance']:<20.4f}"
    )

    print("\nSUBGRAPH COMPARISON")
    print("===================")
    print(
        f"{'Method':<20}"
        f"{'Communities':<15}"
        f"{'Modularity':<15}"
        f"{'Avg Conductance':<20}"
    )

    print(
        f"{'Girvan-Newman':<20}"
        f"{girvan_newman_results['number_of_communities']:<15}"
        f"{girvan_newman_results['modularity']:<15.4f}"
        f"{girvan_newman_results['conductance']:<20.4f}"
    )

    print(
        f"{'Spectral':<20}"
        f"{spectral_small_results['number_of_communities']:<15}"
        f"{spectral_small_results['modularity']:<15.4f}"
        f"{spectral_small_results['conductance']:<20.4f}"
    )


def plot_degree_distribution(graph):
    degrees = [degree for node, degree in graph.degree()]

    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50)

    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")

    plt.savefig(f"{OUTPUT_DIR}/degree_distribution.png", bbox_inches="tight")
    plt.close()


def plot_community_sizes(results, method_name, filename):
    community_sizes = results["community_sizes"]

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(community_sizes) + 1), community_sizes)

    plt.title(f"Community Sizes - {method_name}")
    plt.xlabel("Community")
    plt.ylabel("Number of nodes")

    plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight")
    plt.close()


def plot_modularity_comparison(
    louvain_results, spectral_results, girvan_newman_results, spectral_small_results
):
    methods = [
        "Louvain\nFull",
        "Spectral\nFull",
        "Girvan-Newman\nSubgraph",
        "Spectral\nSubgraph",
    ]

    modularity_scores = [
        louvain_results["modularity"],
        spectral_results["modularity"],
        girvan_newman_results["modularity"],
        spectral_small_results["modularity"],
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, modularity_scores)

    plt.title("Modularity Comparison")
    plt.xlabel("Method")
    plt.ylabel("Modularity")
    plt.ylim(0, 1)

    plt.savefig(f"{OUTPUT_DIR}/modularity_comparison.png", bbox_inches="tight")
    plt.close()


def plot_conductance_comparison(
    louvain_results, spectral_results, girvan_newman_results, spectral_small_results
):
    methods = [
        "Louvain\nFull",
        "Spectral\nFull",
        "Girvan-Newman\nSubgraph",
        "Spectral\nSubgraph",
    ]

    conductance_scores = [
        louvain_results["conductance"],
        spectral_results["conductance"],
        girvan_newman_results["conductance"],
        spectral_small_results["conductance"],
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, conductance_scores)

    plt.title("Average Conductance Comparison")
    plt.xlabel("Method")
    plt.ylabel("Average Conductance")

    plt.savefig(f"{OUTPUT_DIR}/conductance_comparison.png", bbox_inches="tight")
    plt.close()


def plot_small_graph_communities(graph, results, method_name, filename):
    communities = results["communities"]

    node_to_community = {}

    for community_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_id

    node_colors = [node_to_community[node] for node in graph.nodes()]

    position = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(12, 10))

    nx.draw_networkx_nodes(
        graph, position, node_color=node_colors, node_size=40, cmap=plt.cm.tab20
    )

    nx.draw_networkx_edges(graph, position, alpha=0.2, width=0.5)

    plt.title(f"Detected Communities - {method_name}")
    plt.axis("off")

    plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight")
    plt.close()


def main():
    create_output_folder()

    graph = load_graph()

    analyze_graph(graph)
    plot_degree_distribution(graph)

    louvain_results = run_louvain(graph)

    small_graph = get_top_degree_subgraph(graph, n=500)
    small_graph = get_largest_connected_component_subgraph(small_graph)

    display_subgraph_information(small_graph)

    girvan_newman_results = run_girvan_newman(small_graph, k=4)

    spectral_results = run_spectral_clustering(
        graph, k=louvain_results["number_of_communities"]
    )

    spectral_small_results = run_spectral_clustering(small_graph, k=4)

    display_community_detection_comparison(
        louvain_results,
        spectral_results,
        girvan_newman_results,
        spectral_small_results,
    )

    plot_community_sizes(
        louvain_results, "Louvain - Full Graph", "community_sizes_louvain.png"
    )

    plot_community_sizes(
        spectral_results,
        "Spectral Clustering - Full Graph",
        "community_sizes_spectral_full.png",
    )

    plot_community_sizes(
        girvan_newman_results,
        "Girvan-Newman - Subgraph",
        "community_sizes_girvan_newman.png",
    )

    plot_community_sizes(
        spectral_small_results,
        "Spectral Clustering - Subgraph",
        "community_sizes_spectral_subgraph.png",
    )

    plot_modularity_comparison(
        louvain_results, spectral_results, girvan_newman_results, spectral_small_results
    )

    plot_conductance_comparison(
        louvain_results, spectral_results, girvan_newman_results, spectral_small_results
    )

    plot_small_graph_communities(
        small_graph,
        girvan_newman_results,
        "Girvan-Newman Subgraph",
        "small_graph_girvan_newman_communities.png",
    )

    plot_small_graph_communities(
        small_graph,
        spectral_small_results,
        "Spectral Clustering Subgraph",
        "small_graph_spectral_communities.png",
    )


if __name__ == "__main__":
    main()
