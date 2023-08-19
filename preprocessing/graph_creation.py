# Create an empty graph
from typing import List, Any

import networkx as nx
import pandas as pd
import torch


def create_graph_dataset(df: pd.DataFrame):
    """
    :type df: object
    """
    # Create an empty graph
    G = nx.MultiGraph()

    # Add nodes to the graph for each unique card_id, merchant_name
    G.add_nodes_from(df["card_id"].unique(), type="card_id")
    G.add_nodes_from(df["Merchant Name"].unique(), type="merchant_name")

    # The code below adding edges and properties to the edges of a graph.
    # The code iterates through each row of the dataframe, df,
    # and creates a variable for each property then
    # assign it to the edge between the card_id and merchant_name of that row.

    # Add edges and properties to the edges
    for _, row in df.iterrows():
        # Create a variable for each properties for each edge

        year = (row["Year"],)
        month = (row["Month"],)
        day = (row["Day"],)
        hour = (row["Hour"],)
        minute = (row["Minute"],)
        amount = (row["Amount"],)
        use_chip = (row["Use Chip"],)
        merchant_city = (row["Merchant City"],)
        errors = (row["Errors?"],)
        mcc = row["MCC"]

        G.add_edge(
            row["card_id"],
            row["Merchant Name"],
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            amount=amount,
            use_chip=use_chip,
            merchant_city=merchant_city,
            errors=errors,
            mcc=mcc,
        )

    # Get the number of nodes and edges in the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Print the number of nodes and edges
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    # Convert the graph to an adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    print(f"adj_matrix.shape: {adj_matrix.shape}")

    # We define the variable "edge_list" which is a list of edges and their associated data in a graph G.
    # Then we create an empty list called "x" and iterates over each edge in the edge_list.
    # For each edge, it extracts the values of the edge data, converts them to floats if needed, and append them to the list "x".
    # Finally, we convert the list "x" to a PyTorch tensor with float datatype

    # Prepare the data for input into the model
    edge_list = list(G.edges(data=True))
    features = []
    for edge in edge_list:
        edge_values = list(edge[2].values())
        edge_values = [
            float(i[0])
            if type(i) == tuple and type(i[0]) == str
            else i[0]
            if type(i) == tuple
            else i
            for i in edge_values
        ]
        features.append(edge_values)

    print(f"features.shape: {len(features)}")

    if "Is Fraud?" in df:
        target = df["Is Fraud?"].values[:, None].tolist()
        # target = df["Is Fraud?"].values.tolist()
        print(f"target: {len(target)}")
        return features, target
    else:
        return features, None
