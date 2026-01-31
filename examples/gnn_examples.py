"""
Comprehensive Examples: Graph Neural Networks

This file demonstrates all GNN architectures with practical examples and use cases.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
from ilovetools.ml.gnn import (
    GCN,
    GAT,
    GraphSAGE,
    GIN,
    create_adjacency_matrix,
    graph_pooling,
)

print("=" * 80)
print("GRAPH NEURAL NETWORKS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Graph Convolutional Network (GCN) - Node Classification
# ============================================================================
print("EXAMPLE 1: GCN - Node Classification (Citation Network)")
print("-" * 80)

# Simulate citation network (papers citing each other)
num_papers = 100
num_features = 128  # Paper features (e.g., TF-IDF)
num_classes = 7  # Paper categories

gcn = GCN(in_features=num_features, hidden_features=256, out_features=num_classes, num_layers=2)

print("Citation network:")
print(f"Papers: {num_papers}")
print(f"Features per paper: {num_features}")
print(f"Categories: {num_classes}")
print()

# Paper features
node_features = np.random.randn(num_papers, num_features)

# Citation graph (adjacency matrix)
adj_matrix = np.random.randint(0, 2, (num_papers, num_papers))

print(f"Node features: {node_features.shape}")
print(f"Adjacency matrix: {adj_matrix.shape}")
print()

# Forward pass
logits = gcn.forward(node_features, adj_matrix, training=False)

print(f"Output logits: {logits.shape}")
print(f"Predictions: argmax(logits, axis=1)")

# Predict categories
predictions = np.argmax(logits, axis=1)
print(f"Predicted categories for first 10 papers: {predictions[:10]}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Graph Attention Network (GAT) - Social Network
# ============================================================================
print("EXAMPLE 2: GAT - Friend Recommendation (Social Network)")
print("-" * 80)

# Simulate social network
num_users = 200
user_features_dim = 64  # User profile features
num_heads = 8

gat = GAT(in_features=user_features_dim, hidden_features=32, num_heads=num_heads)

print("Social network:")
print(f"Users: {num_users}")
print(f"User features: {user_features_dim}")
print(f"Attention heads: {num_heads}")
print()

# User features (age, interests, activity, etc.)
user_features = np.random.randn(num_users, user_features_dim)

# Friendship graph
friendship_matrix = np.random.randint(0, 2, (num_users, num_users))

print(f"User features: {user_features.shape}")
print(f"Friendship matrix: {friendship_matrix.shape}")
print()

# Forward pass
user_embeddings = gat.forward(user_features, friendship_matrix, training=False)

print(f"User embeddings: {user_embeddings.shape}")
print(f"Each user → {32 * num_heads}D embedding")
print()

print("Use embeddings for:")
print("✓ Friend recommendations (similarity search)")
print("✓ Community detection (clustering)")
print("✓ Influence prediction")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: GraphSAGE - Large-Scale Graph (Inductive Learning)
# ============================================================================
print("EXAMPLE 3: GraphSAGE - Product Recommendations (E-commerce)")
print("-" * 80)

# Simulate product graph
num_products = 1000
product_features_dim = 256
num_samples = 25

sage = GraphSAGE(
    in_features=product_features_dim,
    hidden_features=512,
    num_layers=2,
    aggregator='mean',
    num_samples=num_samples
)

print("Product recommendation graph:")
print(f"Products: {num_products}")
print(f"Product features: {product_features_dim}")
print(f"Neighbors sampled per layer: {num_samples}")
print()

# Product features (category, price, ratings, etc.)
product_features = np.random.randn(num_products, product_features_dim)

# Product similarity graph (co-purchase, co-view)
product_graph = np.random.randint(0, 2, (num_products, num_products))

print(f"Product features: {product_features.shape}")
print(f"Product graph: {product_graph.shape}")
print()

# Forward pass
product_embeddings = sage.forward(product_features, product_graph)

print(f"Product embeddings: {product_embeddings.shape}")
print(f"Scalable to millions of products!")
print()

print("Use embeddings for:")
print("✓ Similar product recommendations")
print("✓ Bundle suggestions")
print("✓ Search ranking")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: GIN - Molecular Property Prediction
# ============================================================================
print("EXAMPLE 4: GIN - Drug Discovery (Molecular Graphs)")
print("-" * 80)

# Simulate molecular graph
num_atoms = 50  # Atoms in molecule
atom_features_dim = 128  # Atom type, charge, etc.

gin = GIN(in_features=atom_features_dim, hidden_features=256, num_layers=5)

print("Molecular graph:")
print(f"Atoms: {num_atoms}")
print(f"Atom features: {atom_features_dim}")
print(f"GIN layers: 5 (high expressiveness)")
print()

# Atom features
atom_features = np.random.randn(num_atoms, atom_features_dim)

# Molecular bonds (adjacency matrix)
bond_matrix = np.random.randint(0, 2, (num_atoms, num_atoms))

print(f"Atom features: {atom_features.shape}")
print(f"Bond matrix: {bond_matrix.shape}")
print()

# Get atom embeddings
atom_embeddings = gin.forward(atom_features, bond_matrix)

print(f"Atom embeddings: {atom_embeddings.shape}")
print()

# Pool to molecule-level
molecule_embedding = graph_pooling(atom_embeddings, method='sum')

print(f"Molecule embedding: {molecule_embedding.shape}")
print()

print("Predict molecular properties:")
print("✓ Toxicity")
print("✓ Solubility")
print("✓ Binding affinity")
print("✓ Drug-likeness")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Creating Graph from Edge List
# ============================================================================
print("EXAMPLE 5: Creating Graph from Edge List")
print("-" * 80)

# Edge list (source, target)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 4), (1, 3), (2, 4)
]

num_nodes = 5

print("Edge list:")
for src, tgt in edges:
    print(f"  {src} → {tgt}")
print()

# Create adjacency matrix
adj_matrix = create_adjacency_matrix(edges, num_nodes=num_nodes)

print(f"Adjacency matrix:\n{adj_matrix}")
print()

# Node features
node_features = np.random.randn(num_nodes, 64)

# Apply GCN
gcn = GCN(in_features=64, hidden_features=128, out_features=32)
output = gcn.forward(node_features, adj_matrix, training=False)

print(f"Node embeddings: {output.shape}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Graph Pooling for Graph Classification
# ============================================================================
print("EXAMPLE 6: Graph Pooling - Molecule Classification")
print("-" * 80)

# Multiple molecules
num_molecules = 10
atoms_per_molecule = 30
atom_features = 128

print(f"Dataset: {num_molecules} molecules")
print(f"Atoms per molecule: {atoms_per_molecule}")
print()

gin = GIN(in_features=atom_features, hidden_features=256, out_features=128)

molecule_embeddings = []

for i in range(num_molecules):
    # Molecule graph
    atoms = np.random.randn(atoms_per_molecule, atom_features)
    bonds = np.random.randint(0, 2, (atoms_per_molecule, atoms_per_molecule))
    
    # Get atom embeddings
    atom_emb = gin.forward(atoms, bonds)
    
    # Pool to molecule-level
    mol_emb = graph_pooling(atom_emb, method='mean')
    molecule_embeddings.append(mol_emb)

molecule_embeddings = np.array(molecule_embeddings)

print(f"Molecule embeddings: {molecule_embeddings.shape}")
print()

print("Pooling methods:")
print("✓ Mean: Average atom features")
print("✓ Max: Maximum activation")
print("✓ Sum: Total molecular features")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Link Prediction
# ============================================================================
print("EXAMPLE 7: Link Prediction - Knowledge Graph Completion")
print("-" * 80)

# Knowledge graph (entities and relations)
num_entities = 150
entity_features_dim = 200

sage = GraphSAGE(in_features=entity_features_dim, hidden_features=256)

print("Knowledge graph:")
print(f"Entities: {num_entities}")
print(f"Entity features: {entity_features_dim}")
print()

# Entity features
entity_features = np.random.randn(num_entities, entity_features_dim)

# Known relations (adjacency matrix)
relation_matrix = np.random.randint(0, 2, (num_entities, num_entities))

print(f"Entity features: {entity_features.shape}")
print(f"Relation matrix: {relation_matrix.shape}")
print()

# Get entity embeddings
entity_embeddings = sage.forward(entity_features, relation_matrix)

print(f"Entity embeddings: {entity_embeddings.shape}")
print()

# Predict link between entity 0 and entity 50
entity_0 = entity_embeddings[0]
entity_50 = entity_embeddings[50]

link_score = np.dot(entity_0, entity_50)

print(f"Link prediction score (entity 0 ↔ entity 50): {link_score:.4f}")
print()

print("Applications:")
print("✓ Knowledge graph completion")
print("✓ Relation extraction")
print("✓ Question answering")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Multi-Layer GCN
# ============================================================================
print("EXAMPLE 8: Multi-Layer GCN - Deep Graph Learning")
print("-" * 80)

num_nodes = 100
input_features = 128

# 3-layer GCN
gcn_deep = GCN(
    in_features=input_features,
    hidden_features=256,
    out_features=64,
    num_layers=3,
    dropout=0.5
)

print("Deep GCN:")
print(f"Layers: 3")
print(f"Architecture: {input_features} → 256 → 256 → 64")
print(f"Dropout: 0.5")
print()

node_features = np.random.randn(num_nodes, input_features)
adj_matrix = np.random.randint(0, 2, (num_nodes, num_nodes))

output = gcn_deep.forward(node_features, adj_matrix, training=True)

print(f"Output: {output.shape}")
print()

print("Benefits of deeper GCNs:")
print("✓ Larger receptive field")
print("✓ More complex patterns")
print("✓ Better expressiveness")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Comparing GNN Architectures
# ============================================================================
print("EXAMPLE 9: Comparing GNN Architectures")
print("-" * 80)

num_nodes = 100
features = 128

node_features = np.random.randn(num_nodes, features)
adj_matrix = np.random.randint(0, 2, (num_nodes, num_nodes))

print(f"Graph: {num_nodes} nodes, {features} features")
print()

# GCN
gcn = GCN(features, 256, 64)
gcn_out = gcn.forward(node_features, adj_matrix, training=False)

# GAT
gat = GAT(features, 32, num_heads=8)
gat_out = gat.forward(node_features, adj_matrix, training=False)

# GraphSAGE
sage = GraphSAGE(features, 256)
sage_out = sage.forward(node_features, adj_matrix)

# GIN
gin = GIN(features, 256, num_layers=5)
gin_out = gin.forward(node_features, adj_matrix)

print("Output shapes:")
print(f"GCN: {gcn_out.shape}")
print(f"GAT: {gat_out.shape}")
print(f"GraphSAGE: {sage_out.shape}")
print(f"GIN: {gin_out.shape}")
print()

print("When to use:")
print("✓ GCN: General node classification, semi-supervised")
print("✓ GAT: Varying neighbor importance, attention needed")
print("✓ GraphSAGE: Large graphs, inductive learning")
print("✓ GIN: Graph classification, high expressiveness")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Traffic Network - Route Prediction
# ============================================================================
print("EXAMPLE 10: Traffic Network - Route Prediction")
print("-" * 80)

# Road network
num_intersections = 200
intersection_features_dim = 64  # Traffic flow, time, etc.

gcn = GCN(in_features=intersection_features_dim, hidden_features=128, out_features=32)

print("Traffic network:")
print(f"Intersections: {num_intersections}")
print(f"Features: {intersection_features_dim}")
print()

# Intersection features (traffic density, avg speed, etc.)
intersection_features = np.random.randn(num_intersections, intersection_features_dim)

# Road connections
road_network = np.random.randint(0, 2, (num_intersections, num_intersections))

print(f"Intersection features: {intersection_features.shape}")
print(f"Road network: {road_network.shape}")
print()

# Get intersection embeddings
intersection_embeddings = gcn.forward(intersection_features, road_network, training=False)

print(f"Intersection embeddings: {intersection_embeddings.shape}")
print()

print("Applications:")
print("✓ Traffic prediction")
print("✓ Route optimization")
print("✓ Congestion detection")
print("✓ Travel time estimation")

print("\n✓ Example 10 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ GCN - Node Classification (Citation Network)")
print("2. ✓ GAT - Friend Recommendation (Social Network)")
print("3. ✓ GraphSAGE - Product Recommendations")
print("4. ✓ GIN - Drug Discovery (Molecular Graphs)")
print("5. ✓ Creating Graph from Edge List")
print("6. ✓ Graph Pooling - Molecule Classification")
print("7. ✓ Link Prediction - Knowledge Graph")
print("8. ✓ Multi-Layer GCN")
print("9. ✓ Comparing GNN Architectures")
print("10. ✓ Traffic Network - Route Prediction")
print()
print("You now have a complete understanding of Graph Neural Networks!")
print()
print("Next steps:")
print("- Use GCN for node classification")
print("- Use GAT when neighbor importance varies")
print("- Use GraphSAGE for large-scale graphs")
print("- Use GIN for graph classification")
print("- Apply to your domain (social, molecular, traffic, etc.)")
