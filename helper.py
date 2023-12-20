import os
import sys
import random
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite, centrality, community
import matplotlib.pyplot as plt


def print_network_information(G):
    print(f"Information for given Graph with name '{G.name}':")
    print(f"\tGraph is directed: {G.is_directed()}")
    print(f"\tNumber of nodes: {G.number_of_nodes()}")
    print(f"\tNumber of edges: {G.number_of_edges()}")


def camel_case():
    print("camel case")


def draw_emphasizing_edge_weight(G, edge_weight_scaling_factor=None):
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

    # Calculate the maximum weight to normalize the edge widths
    max_weight = max(edge[2] for edge in G.edges(data='weight'))
    
    # If a scaling factor is not provided, calculate a default one
    if not edge_weight_scaling_factor:
        edge_weight_scaling_factor = 10.0 / max_weight

    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], 
                               width=edge[2]*edge_weight_scaling_factor)


def draw_graph_and_color_groups(G, groups=None, layout="circular"):
    """
    At most 8 different groups supported. If groups is None all nodes belong to the same group
    """
    if groups is None:
        groups = [G.nodes()]

    pos = None
    if (layout == "circular"):
        pos = nx.drawing.layout.circular_layout(G)
    elif (layout == "spring"):
        pos = nx.drawing.layout.spring_layout(G)
    color = ['lightblue', 'lightgreen', 'tan', 'orange', 'lavender', 'grey', 'magenta', 'beige']

    for i, group in enumerate(groups):
        nx.draw_networkx_nodes(G, pos, nodelist=group, node_color=color[i])

    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')


def draw_graph_with_centralities(G, centrality_map):
    plt.figure()
    pos_nodes = nx.spring_layout(G)
    nx.draw(G, pos_nodes, node_color="lightblue", with_labels=False,
            node_size=[(centrality_map[node] + 1) * 100 for node in G.nodes()])

    labels = {}
    for k, v in centrality_map.items():
        labels[k] = f"{k} ({centrality_map.get(k):.3f})"

    nx.draw_networkx_labels(G, pos_nodes, labels=labels)
    plt.show()
    

def perform_random_walk(graph, starting_node, num_steps):
    """
    Perform a random walk on a bipartite graph, starting from a given node,
    and alternating between the two types of nodes (heroes and comics).
    """    
    sampled_nodes = set([starting_node])
    current_node = starting_node

    for _ in range(num_steps):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            current_node = random.choice(neighbors)
            sampled_nodes.add(current_node)
    
    return sampled_nodes


def print_centrality(centrality, title):
    """
    Print the top 5 nodes by centrality, along with their centrality value.
    """
    print(f"{title}:")
    for hero, value in centrality[:5]:
        print(f"  {hero}: {value:.4f}")
    print()


def get_communities(G):
    """
    Get the communities in the graph using the Greedy Modularity algorithm
    """
    # Use the Greedy Modularity Community detection algorithm
    communities = community.greedy_modularity_communities(G)

    # Output the communities
    for i, comm in enumerate(communities):
        print(f"Gemeinschaft {i+1}: {sorted(comm)[:10]} ... [{len(comm)} Helden]")  # Show first 10 heroes
        if i >= 4:  # Limit to showing details for the first 5 communities
            break
    
    print()

    # Total number of communities detected
    total_communities = len(communities)
    print(f"Total number of communities detected: {total_communities}")

    return communities


def plot_community(G, communities, community_number):
    """
    Plot the given community
    """
    community = communities[community_number]

    community_subgraph = G.subgraph(community)

    G_community = nx.Graph()

    # Add nodes to the graph
    G_community.add_nodes_from(community_subgraph)

    # Iterate over all pairs of nodes and add an edge with a weight equal to the number of shared comics
    for hero1 in community_subgraph:
        for hero2 in community_subgraph:
            if hero1 != hero2:
                # The number of shared comics is the number of common neighbors in the one-mode network
                shared_comics = len(set(G.neighbors(hero1)) & set(G.neighbors(hero2)))
                if shared_comics > 0:
                    # Add an edge with a weight equal to the number of shared comics
                    G_community.add_edge(hero1, hero2, weight=shared_comics)

    plt.figure(figsize=(20, 10))
    plt.title(f"Community {community_number + 1} Network ({len(G_community.nodes())} Nodes, {len(G_community.edges())} Edges)", fontsize=16)
    draw_emphasizing_edge_weight(G_community)
    plt.show()

    print(f"Top 10 Kantengewichte der Helden in der Community {community_number + 1}:")
    for edge in sorted(G_community.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]:
        print(f"{edge[0]} - {edge[1]}: {edge[2]['weight']}")


def plot_community_edge_betweenness(G, communities, community_number):
    """
    Plot a specific community from the graph with edge betweenness centrality.

    Parameters:
    G (NetworkX graph): The one-mode graph.
    communities (list): A list of communities (sets of nodes).
    index (int): The index of the community to plot.
    """

    community = communities[community_number]

    community_subgraph = G.subgraph(community)

    # Calculate edge betweenness centrality for the community subgraph
    edge_betweenness = nx.edge_betweenness_centrality(community_subgraph)

    plt.figure(figsize=(20, 10))
    plt.title(f"Community {community_number+1} with Edge Betweenness Centrality", fontsize=16)

    pos = nx.spring_layout(community_subgraph, seed=42)

    # Draw the nodes
    nx.draw_networkx_nodes(community_subgraph, pos, node_color="lightblue", node_size=100)

    # Draw the edges with width proportional to the edge betweenness centrality
    for edge, eb in edge_betweenness.items():
        nx.draw_networkx_edges(community_subgraph, pos, edgelist=[edge], width=eb*10)  # Scale the width for visibility

    # Draw the labels
    nx.draw_networkx_labels(community_subgraph, pos, font_size=12)

    plt.show()

    # print the top 10 edges by edge betweenness centrality
    print(f"Top 10 Edges by Edge Betweenness Centrality in Community {community_number+1}:")
    for edge in sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{edge[0]}: {edge[1]:.4f}")