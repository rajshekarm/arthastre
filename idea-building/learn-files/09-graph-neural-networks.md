# 09 - Graph Neural Networks

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Completed
- [ ] Reviewed

**Time Estimate:** 3-4 weeks  
**Started:** [Date]  
**Completed:** [Date]

---

## What I Need to Learn

Understanding and implementing Graph Neural Networks (GNNs) to learn patterns from my graph structure and create meaningful node embeddings.

---

## Prerequisites

- [ ] Completed [07-embeddings.md](07-embeddings.md)
- [ ] Completed [08-neural-networks-basics.md](08-neural-networks-basics.md)
- [ ] Understand PyTorch basics
- [ ] Have graph database with relationships

---

## The Core Idea

**Traditional ML:** Nodes are independent
**GNN:** Nodes learn from their neighbors

```
Node's embedding = f(
    its own features,
    neighbors' features,
    relationship types
)
```

**My intuition:**
[How I understand this concept]

---

## Message Passing

### The Key Mechanism

```python
def message_passing_round(graph):
    """
    Each node aggregates information from neighbors
    """
    for node in graph.nodes:
        # Step 1: Collect messages from neighbors
        messages = []
        for neighbor in node.neighbors:
            # Transform neighbor's features
            message = W * neighbor.embedding  # W is learned weight matrix
            messages.append(message)
        
        # Step 2: Aggregate (sum, mean, max)
        aggregated = sum(messages)  # or mean(messages)
        
        # Step 3: Update node's embedding
        node.embedding = activation(
            node.embedding + aggregated
        )
```

**After multiple rounds:**
- Round 1: Node knows about 1-hop neighbors
- Round 2: Node knows about 2-hop neighbors
- Round 3: Node knows about 3-hop neighbors

**My understanding:**
[Explain in my own words]

---

## My First GNN

### Using PyTorch Geometric

#### Step 1: Install
```bash
pip install torch-geometric torch-scatter torch-sparse
```

#### Step 2: Prepare Data
```python
import torch
from torch_geometric.data import Data

# Convert my Neo4j graph to PyG format
def neo4j_to_pyg(neo4j_graph):
    """
    Convert Neo4j graph to PyTorch Geometric format
    """
    # Get all nodes
    nodes = neo4j_graph.run("MATCH (n:Company) RETURN n").data()
    
    # Create node feature matrix
    node_features = []
    node_mapping = {}
    
    for idx, node in enumerate(nodes):
        node_mapping[node['ticker']] = idx
        # Features: [market_cap, sector_encoded, ...]
        features = [
            node['market_cap'] / 1e12,  # Normalize
            encode_sector(node['sector']),
            # ... more features
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Get all edges
    edges = neo4j_graph.run("""
        MATCH (a:Company)-[r]->(b:Company)
        RETURN a.ticker, b.ticker, type(r), r.weight
    """).data()
    
    # Create edge index
    edge_index = []
    edge_attr = []
    
    for edge in edges:
        from_idx = node_mapping[edge['a.ticker']]
        to_idx = node_mapping[edge['b.ticker']]
        edge_index.append([from_idx, to_idx])
        edge_attr.append(edge['r.weight'])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_mapping
```

**My conversion code:**
```python
[My actual implementation]
```

---

#### Step 3: Build GNN Model

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MyFirstGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(MyFirstGNN, self).__init__()
        
        # Two GCN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        
        return x

# Create model
model = MyFirstGNN(
    num_features=10,  # My input feature size
    hidden_dim=64,
    output_dim=32     # Embedding dimension
)
```

**My model:**
```python
[My GNN architecture]
```

---

#### Step 4: Train the Model

```python
def train_gnn(model, data, epochs=200):
    """
    Train GNN for node classification or link prediction
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data)
        
        # Loss depends on task
        # For node classification:
        loss = F.cross_entropy(embeddings[train_mask], labels[train_mask])
        
        # For link prediction:
        # loss = link_prediction_loss(embeddings, pos_edges, neg_edges)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return model

# Train
trained_model = train_gnn(model, graph_data)
```

**My training loop:**
```python
[My implementation]
```

---

#### Step 5: Generate Embeddings

```python
def get_embeddings(model, data):
    """
    Get node embeddings from trained model
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
    
    return embeddings.numpy()

# Get embeddings for all nodes
node_embeddings = get_embeddings(trained_model, graph_data)

# Now I can:
# 1. Find similar companies
# 2. Cluster companies
# 3. Predict missing relationships
# 4. Classify nodes
```

**My embeddings:**
```python
[How I use the embeddings]
```

---

## Use Cases I'm Implementing

### Use Case 1: Find Similar Companies

```python
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_companies(company_ticker, node_embeddings, node_mapping, top_k=5):
    """
    Find companies with similar embeddings
    """
    # Get company's embedding
    company_idx = node_mapping[company_ticker]
    company_emb = node_embeddings[company_idx].reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(company_emb, node_embeddings)[0]
    
    # Get top-k (excluding itself)
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    
    # Map back to tickers
    idx_to_ticker = {v: k for k, v in node_mapping.items()}
    similar_companies = [
        (idx_to_ticker[idx], similarities[idx])
        for idx in top_indices
    ]
    
    return similar_companies

# Example
similar = find_similar_companies('NVDA', node_embeddings, node_mapping)
# Result: [('AMD', 0.92), ('INTC', 0.87), ('QCOM', 0.81), ...]
```

**My results:**
```
NVDA similar to:
1. [Ticker] - similarity: [score]
2. [Ticker] - similarity: [score]
...
```

---

### Use Case 2: Link Prediction

```python
def predict_missing_links(model, data, node_mapping, threshold=0.7):
    """
    Predict which companies should be connected
    """
    embeddings = get_embeddings(model, data)
    
    # Calculate similarity between all pairs
    similarities = cosine_similarity(embeddings)
    
    # Find high similarity pairs not currently connected
    predictions = []
    for i in range(len(similarities)):
        for j in range(i+1, len(similarities)):
            if similarities[i][j] > threshold:
                # Check if edge exists
                if not edge_exists(data, i, j):
                    ticker_i = get_ticker(node_mapping, i)
                    ticker_j = get_ticker(node_mapping, j)
                    predictions.append((
                        ticker_i,
                        ticker_j,
                        similarities[i][j]
                    ))
    
    return predictions

# Find potential relationships
potential_links = predict_missing_links(model, graph_data, node_mapping)
```

**My predictions:**
```
Suggested relationships:
1. [Company A] <-> [Company B] (confidence: [score])
2. [Company C] <-> [Company D] (confidence: [score])
...
```

---

### Use Case 3: Node Classification

```python
def classify_nodes(embeddings, labels_known, mask_unknown):
    """
    Classify nodes based on embeddings
    Example: Predict sector for companies with unknown sector
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Train classifier on known labels
    clf = RandomForestClassifier()
    clf.fit(embeddings[~mask_unknown], labels_known)
    
    # Predict unknown labels
    predictions = clf.predict(embeddings[mask_unknown])
    
    return predictions
```

---

## Different GNN Architectures

### GCN (Graph Convolutional Network)
```python
from torch_geometric.nn import GCNConv

# Simple, good starting point
# Best for: Homogeneous graphs
```

### GAT (Graph Attention Network)
```python
from torch_geometric.nn import GATConv

# Uses attention mechanism
# Best for: When some neighbors more important than others
```

### GraphSAGE
```python
from torch_geometric.nn import SAGEConv

# Samples neighbors
# Best for: Large graphs, inductive learning
```

### GIN (Graph Isomorphism Network)
```python
from torch_geometric.nn import GINConv

# More expressive
# Best for: Graph classification tasks
```

**I'm using:** [Which architecture]
**Because:** [My reasoning]

---

## Hyperparameter Tuning

### What I'm tuning:

```python
hyperparameters = {
    'hidden_dim': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout': [0.3, 0.5, 0.7],
    'aggregation': ['mean', 'sum', 'max']
}
```

**My best configuration:**
```
hidden_dim: [value]
num_layers: [value]
learning_rate: [value]
dropout: [value]
```

**Results:**
- Training loss: [value]
- Validation accuracy: [value]
- Test performance: [value]

---

## Visualization

### t-SNE Visualization of Embeddings

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, node_mapping, labels):
    """
    Visualize node embeddings in 2D
    """
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6
    )
    
    # Add labels for some nodes
    for ticker, idx in node_mapping.items():
        if ticker in ['NVDA', 'AMD', 'TSLA', 'AAPL']:  # Major companies
            plt.annotate(
                ticker,
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1])
            )
    
    plt.colorbar(scatter, label='Sector')
    plt.title('Company Embeddings Visualization')
    plt.show()
```

**My visualization:**
[Describe what I see - clusters, patterns]

---

## Saving and Loading

```python
# Save trained model
torch.save(model.state_dict(), 'my_gnn_model.pt')

# Save embeddings
np.save('node_embeddings.npy', node_embeddings)
np.save('node_mapping.npy', node_mapping)

# Load model
model = MyFirstGNN(num_features=10, hidden_dim=64, output_dim=32)
model.load_state_dict(torch.load('my_gnn_model.pt'))
model.eval()

# Load embeddings
embeddings = np.load('node_embeddings.npy')
mapping = np.load('node_mapping.npy', allow_pickle=True).item()
```

---

## Integration with Neo4j

### Storing Embeddings Back in Graph

```python
def store_embeddings_in_neo4j(embeddings, node_mapping, neo4j_graph):
    """
    Store GNN embeddings as node properties
    """
    for ticker, idx in node_mapping.items():
        embedding = embeddings[idx].tolist()
        
        query = """
        MATCH (c:Company {ticker: $ticker})
        SET c.gnn_embedding = $embedding
        SET c.embedding_updated = timestamp()
        """
        
        neo4j_graph.run(query, ticker=ticker, embedding=embedding)

# Store embeddings
store_embeddings_in_neo4j(node_embeddings, node_mapping, graph_db)
```

**Now I can query:**
```cypher
// Find similar companies using embeddings
MATCH (c:Company {ticker: 'NVDA'})
MATCH (other:Company)
WHERE c <> other
RETURN other.ticker, 
       gds.similarity.cosine(c.gnn_embedding, other.gnn_embedding) AS similarity
ORDER BY similarity DESC
LIMIT 5
```

---

## Challenges I Faced

### Challenge 1: [Problem]
**Solution:** [How I solved it]

### Challenge 2: [Problem]
**Solution:** [How I solved it]

---

## Performance Metrics

### Embedding Quality
```python
# Measure how well embeddings capture graph structure

# 1. Link prediction accuracy
# 2. Node classification accuracy
# 3. Clustering quality (silhouette score)
# 4. Nearest neighbor precision
```

**My metrics:**
- Link prediction AUC: [score]
- Node classification F1: [score]
- Clustering quality: [score]

---

## Next Steps

- [ ] Complete [10-multi-modal-fusion.md](10-multi-modal-fusion.md)
- [ ] Experiment with different architectures
- [ ] Tune hyperparameters
- [ ] Deploy embeddings to production

---

## Resources I Used

- [ ] PyTorch Geometric Tutorial: [Link]
- [ ] Stanford CS224W: [Notes]
- [ ] Paper: [Which paper]

---

## My Notes

[Breakthroughs, confusions cleared up, things to remember]

---

**Status:** [Date] - [Progress]
