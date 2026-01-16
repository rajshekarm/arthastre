# 05 - Graph Algorithms

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Completed
- [ ] Reviewed

**Time Estimate:** 2 weeks  
**Started:** [Date]  
**Completed:** [Date]

---

## What I Need to Learn

Understanding and implementing graph algorithms to extract insights from my financial network. From basic traversal to advanced centrality measures and community detection.

---

## Prerequisites

- [ ] Completed [01-graph-basics.md](01-graph-basics.md)
- [ ] Completed [02-graph-database.md](02-graph-database.md)
- [ ] Have working Neo4j database with relationships

---

## Algorithm Categories

### 1. Pathfinding & Traversal
### 2. Centrality & Importance
### 3. Community Detection
### 4. Link Analysis
### 5. Similarity & Recommendation

---

## PART 1: Pathfinding & Traversal

### 1.1 Breadth-First Search (BFS)

**What it does:** Explore graph level by level from starting node.

**Use case:** Find shortest path between two companies.

#### Cypher Implementation
```cypher
// Find shortest path between NVIDIA and Apple
MATCH path = shortestPath(
  (nvda:Company {ticker: 'NVDA'})-[*]-(aapl:Company {ticker: 'AAPL'})
)
RETURN path
```

#### Python + NetworkX
```python
import networkx as nx

def bfs_shortest_path(graph, start, end):
    """
    Find shortest path using BFS
    """
    try:
        path = nx.shortest_path(graph, start, end)
        return path
    except nx.NetworkXNoPath:
        return None

# Example
G = nx.Graph()
G.add_edges_from([
    ('NVDA', 'TSLA'),
    ('TSLA', 'AAPL'),
    ('NVDA', 'AMD')
])

path = bfs_shortest_path(G, 'NVDA', 'AAPL')
print(f"Path: {path}")  # ['NVDA', 'TSLA', 'AAPL']
```

#### 📝 My Practice

**Exercise 1:** Find path between two companies in my graph
```cypher
// My query:
[Write my actual query here]

// Result:
[Document the path I found]
```

**Exercise 2:** Find ALL paths (not just shortest)
```cypher
MATCH path = (a:Company {ticker: 'NVDA'})-[*1..3]-(b:Company {ticker: 'AAPL'})
RETURN path
LIMIT 10
```

**My result:**
```
[List all paths I found]
```

---

### 1.2 Depth-First Search (DFS)

**What it does:** Explore as deep as possible before backtracking.

**Use case:** Detect cycles, explore supplier chains.

#### Python Implementation
```python
def dfs_explore(graph, start, visited=None):
    """
    Explore graph depth-first
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(f"Visiting: {start}")
    
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            dfs_explore(graph, neighbor, visited)
    
    return visited

# Example
visited_nodes = dfs_explore(G, 'NVDA')
print(f"Visited {len(visited_nodes)} nodes")
```

#### 📝 My Practice

**Exercise 3:** Explore supply chain depth-first
```python
# My implementation:
[My DFS code]

# Companies I visited in order:
[List them]
```

---

### 1.3 Dijkstra's Algorithm (Weighted Shortest Path)

**What it does:** Find shortest path considering edge weights.

**Use case:** Find strongest connection path (using relationship weights).

#### Cypher with Weights
```cypher
// Find path with highest cumulative weight
CALL gds.shortestPath.dijkstra.stream('myGraph', {
    sourceNode: id(nvda),
    targetNode: id(aapl),
    relationshipWeightProperty: 'weight'
})
YIELD path, totalCost
RETURN path, totalCost
```

#### NetworkX Implementation
```python
def weighted_shortest_path(graph, start, end, weight='weight'):
    """
    Find shortest path considering weights
    """
    try:
        path = nx.dijkstra_path(graph, start, end, weight=weight)
        length = nx.dijkstra_path_length(graph, start, end, weight=weight)
        return path, length
    except nx.NetworkXNoPath:
        return None, float('inf')

# Example with weighted edges
G = nx.Graph()
G.add_edge('NVDA', 'TSLA', weight=0.87)
G.add_edge('TSLA', 'AAPL', weight=0.65)
G.add_edge('NVDA', 'AAPL', weight=0.45)

path, cost = weighted_shortest_path(G, 'NVDA', 'AAPL')
print(f"Path: {path}, Cost: {cost}")
```

#### 📝 My Practice

**Exercise 4:** Find strongest connection path
```python
# My weighted graph:
[My edges with weights]

# Query result:
[Path and total weight]
```

**TODO:** Find the 3 strongest paths between NVIDIA and Microsoft
```python
# My implementation:
[Code here]

# Results:
# Path 1: [companies] - strength: [score]
# Path 2: [companies] - strength: [score]
# Path 3: [companies] - strength: [score]
```

---

## PART 2: Centrality & Importance

### 2.1 Degree Centrality

**What it measures:** How many connections does a node have?

**Use case:** Find most connected companies.

#### Cypher
```cypher
// Find companies by number of connections
MATCH (c:Company)
RETURN c.ticker, 
       size((c)--()) AS degree
ORDER BY degree DESC
LIMIT 10
```

#### NetworkX
```python
def calculate_degree_centrality(graph):
    """
    Calculate degree centrality for all nodes
    """
    centrality = nx.degree_centrality(graph)
    
    # Sort by centrality
    sorted_nodes = sorted(
        centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_nodes

# Example
centrality_scores = calculate_degree_centrality(G)
for node, score in centrality_scores[:5]:
    print(f"{node}: {score:.3f}")
```

#### 📝 My Practice

**Exercise 5:** Find top 10 most connected companies
```cypher
// My query:
[My query here]
```

**My results:**
| Rank | Company | Degree | My Interpretation |
|------|---------|--------|-------------------|
| 1 | [Ticker] | [#] | [Why is it highly connected?] |
| 2 | [Ticker] | [#] | |
| ... |

**TODO:** Calculate both in-degree and out-degree
```cypher
// In-degree (suppliers)
MATCH (c:Company)<-[:SUPPLIES_TO]-(supplier)
RETURN c.ticker, count(supplier) AS in_degree
ORDER BY in_degree DESC

// Out-degree (customers)
MATCH (c:Company)-[:SUPPLIES_TO]->(customer)
RETURN c.ticker, count(customer) AS out_degree
ORDER BY out_degree DESC
```

**My analysis:**
[Which companies have more suppliers vs customers?]

---

### 2.2 PageRank

**What it measures:** Importance based on connections to important nodes.

**Use case:** Find most influential companies in network.

**The idea:** Being connected to important nodes makes you important.

#### Cypher (Neo4j GDS)
```cypher
// Create graph projection
CALL gds.graph.project(
  'myGraph',
  'Company',
  {
    SUPPLIES_TO: {orientation: 'NATURAL'},
    COMPETES_WITH: {orientation: 'UNDIRECTED'}
  }
)

// Run PageRank
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).ticker AS ticker, score
ORDER BY score DESC
LIMIT 20
```

#### NetworkX
```python
def calculate_pagerank(graph, alpha=0.85):
    """
    Calculate PageRank scores
    """
    pagerank = nx.pagerank(graph, alpha=alpha)
    
    sorted_nodes = sorted(
        pagerank.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_nodes

# Example
pr_scores = calculate_pagerank(G)
for node, score in pr_scores:
    print(f"{node}: {score:.4f}")
```

#### 📝 My Practice

**Exercise 6:** Calculate PageRank for my graph
```python
# My implementation:
[My code]
```

**My top 10 by PageRank:**
| Rank | Company | PageRank Score | Degree | Notes |
|------|---------|----------------|--------|-------|
| 1 | | | | |
| 2 | | | | |

**TODO:** Compare PageRank vs simple degree centrality
```python
# Are they different? Why?
[My analysis]
```

**Question I need to answer:**
- Which companies have high PageRank but low degree? (connected to important nodes)
- Which have high degree but low PageRank? (connected to many unimportant nodes)

---

### 2.3 Betweenness Centrality

**What it measures:** How often a node appears on shortest paths between other nodes.

**Use case:** Find "bridge" companies that connect different parts of network.

#### NetworkX
```python
def calculate_betweenness(graph):
    """
    Find nodes that bridge different communities
    """
    betweenness = nx.betweenness_centrality(graph)
    
    sorted_nodes = sorted(
        betweenness.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_nodes

# Example
between_scores = calculate_betweenness(G)
print("Top bridge nodes:")
for node, score in between_scores[:5]:
    print(f"{node}: {score:.3f}")
```

#### Cypher
```cypher
CALL gds.betweenness.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).ticker AS ticker, score
ORDER BY score DESC
LIMIT 10
```

#### 📝 My Practice

**Exercise 7:** Find bridge companies
```python
# My results:
[Companies with high betweenness]
```

**TODO:** Visualize why these are bridges
```
[Draw or describe the network position of these companies]
```

**My interpretation:**
[Why are these companies important connectors?]

---

### 2.4 Closeness Centrality

**What it measures:** How close a node is to all other nodes on average.

**Use case:** Find companies with fastest access to entire network.

#### NetworkX
```python
def calculate_closeness(graph):
    """
    Find nodes closest to all others
    """
    closeness = nx.closeness_centrality(graph)
    
    sorted_nodes = sorted(
        closeness.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_nodes
```

#### 📝 My Practice

**Exercise 8:** Calculate closeness centrality
```python
# My implementation:
[Code]

# Top 5 companies:
[Results and why they're central]
```

---

## PART 3: Community Detection

### 3.1 Louvain Algorithm

**What it does:** Find groups of nodes that are densely connected internally.

**Use case:** Discover industry clusters, sectors.

#### NetworkX
```python
import networkx.algorithms.community as nx_comm

def detect_communities_louvain(graph):
    """
    Find communities using Louvain algorithm
    """
    communities = nx_comm.louvain_communities(graph)
    
    # Print communities
    for idx, community in enumerate(communities):
        print(f"Community {idx}: {community}")
    
    return communities

# Example
communities = detect_communities_louvain(G)
print(f"Found {len(communities)} communities")
```

#### Cypher (Neo4j GDS)
```cypher
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).ticker AS ticker,
       communityId
ORDER BY communityId
```

#### 📝 My Practice

**Exercise 9:** Find communities in my graph
```python
# My results:
[List communities]
```

**My community analysis:**
| Community | Companies | Theme/Sector | Why grouped? |
|-----------|-----------|--------------|--------------|
| 0 | [List] | [Theme] | [Reasoning] |
| 1 | [List] | [Theme] | [Reasoning] |

**TODO:** Visualize communities with colors
```python
[Code to color nodes by community]
```

---

### 3.2 Label Propagation

**What it does:** Fast community detection - nodes adopt labels from neighbors.

**Use case:** Quick clustering of large graphs.

#### NetworkX
```python
def label_propagation(graph):
    """
    Fast community detection
    """
    communities = nx_comm.label_propagation_communities(graph)
    return list(communities)
```

#### 📝 My Practice

**Exercise 10:** Compare Louvain vs Label Propagation
```python
# Run both algorithms
louvain_comms = detect_communities_louvain(G)
label_comms = label_propagation(G)

# Compare results:
[Are they similar? Different? Why?]
```

---

### 3.3 Connected Components

**What it does:** Find disconnected subgraphs.

**Use case:** Identify isolated clusters.

#### NetworkX
```python
def find_connected_components(graph):
    """
    Find all connected components
    """
    components = list(nx.connected_components(graph))
    
    for idx, component in enumerate(components):
        print(f"Component {idx}: {len(component)} nodes")
        print(f"  Nodes: {component}")
    
    return components
```

#### Cypher
```cypher
CALL gds.wcc.stream('myGraph')
YIELD nodeId, componentId
RETURN componentId, 
       collect(gds.util.asNode(nodeId).ticker) AS companies,
       count(*) AS size
ORDER BY size DESC
```

#### 📝 My Practice

**Exercise 11:** Check if my graph is fully connected
```python
# My results:
[Number of components and their sizes]
```

**TODO:** If disconnected, investigate why
```
[Which companies are isolated? Why?]
```

---

## PART 4: Link Analysis & Prediction

### 4.1 Jaccard Similarity

**What it measures:** Similarity based on common neighbors.

**Use case:** Predict potential partnerships.

#### Python
```python
def jaccard_similarity(graph, node1, node2):
    """
    Calculate Jaccard similarity between two nodes
    """
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    
    intersection = len(neighbors1 & neighbors2)
    union = len(neighbors1 | neighbors2)
    
    return intersection / union if union > 0 else 0

# Find companies similar to NVIDIA
def find_similar_companies(graph, company, threshold=0.3):
    """
    Find companies with similar connections
    """
    similarities = []
    
    for other in graph.nodes():
        if other != company:
            sim = jaccard_similarity(graph, company, other)
            if sim > threshold:
                similarities.append((other, sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Example
similar = find_similar_companies(G, 'NVDA')
for company, sim in similar[:5]:
    print(f"{company}: {sim:.3f}")
```

#### 📝 My Practice

**Exercise 12:** Find companies similar to NVIDIA
```python
# My results:
[List of similar companies with scores]
```

**TODO:** Predict missing relationships
```python
# High Jaccard but no direct connection = potential link
[List potential partnerships I discovered]
```

---

### 4.2 Common Neighbors

**What it does:** Count shared neighbors.

**Use case:** Simple link prediction.

#### Cypher
```cypher
// Find potential partnerships based on common customers
MATCH (a:Company {ticker: 'NVDA'})-[:SUPPLIES_TO]->(common)<-[:SUPPLIES_TO]-(b:Company)
WHERE a <> b
RETURN b.ticker, 
       count(common) AS common_customers
ORDER BY common_customers DESC
LIMIT 10
```

#### 📝 My Practice

**Exercise 13:** Find companies with most common customers
```cypher
// My query:
[Query]

// Results:
[Companies and their common customer count]
```

---

### 4.3 Adamic-Adar Index

**What it measures:** Weighted common neighbors (rare neighbors weighted higher).

**Use case:** Better link prediction than simple common neighbors.

#### NetworkX
```python
def adamic_adar_scores(graph, node):
    """
    Calculate Adamic-Adar scores for potential links
    """
    predictions = nx.adamic_adar_index(
        graph,
        [(node, other) for other in graph.nodes() if other != node]
    )
    
    scores = [(node2, score) for _, node2, score in predictions]
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

#### 📝 My Practice

**Exercise 14:** Predict links using Adamic-Adar
```python
# My implementation:
[Code]

# Top predictions:
[Which relationships are likely to form?]
```

---

## PART 5: Similarity & Recommendation

### 5.1 Personalized PageRank

**What it does:** PageRank from a specific starting node's perspective.

**Use case:** Recommend companies based on similarity to a target company.

#### NetworkX
```python
def personalized_pagerank(graph, source_node):
    """
    Calculate PageRank personalized to a source node
    """
    personalization = {node: 0 for node in graph.nodes()}
    personalization[source_node] = 1
    
    ppr = nx.pagerank(graph, personalization=personalization)
    
    # Sort and remove source
    sorted_ppr = sorted(
        [(node, score) for node, score in ppr.items() if node != source_node],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_ppr

# Example: Companies similar to NVIDIA
recommendations = personalized_pagerank(G, 'NVDA')
print("Similar to NVIDIA:")
for company, score in recommendations[:5]:
    print(f"  {company}: {score:.4f}")
```

#### 📝 My Practice

**Exercise 15:** Get recommendations for each major company
```python
# NVIDIA similar companies:
[Results]

# Tesla similar companies:
[Results]

# Pattern I notice:
[My observation]
```

---

### 5.2 Node Similarity

**What it does:** Calculate similarity between nodes based on properties and structure.

#### Cypher (Neo4j GDS)
```cypher
CALL gds.nodeSimilarity.stream('myGraph')
YIELD node1, node2, similarity
WHERE similarity > 0.5
RETURN gds.util.asNode(node1).ticker AS company1,
       gds.util.asNode(node2).ticker AS company2,
       similarity
ORDER BY similarity DESC
LIMIT 20
```

#### 📝 My Practice

**Exercise 16:** Find most similar company pairs
```cypher
// My query:
[Query]

// Surprising similarities I found:
[Companies I didn't expect to be similar]
```

---

## PART 6: Advanced Algorithms

### 6.1 Minimum Spanning Tree

**What it does:** Find minimum cost tree connecting all nodes.

**Use case:** Optimize supply chain connections.

#### NetworkX
```python
def minimum_spanning_tree(graph):
    """
    Find MST of weighted graph
    """
    mst = nx.minimum_spanning_tree(graph, weight='weight')
    
    print(f"Original edges: {graph.number_of_edges()}")
    print(f"MST edges: {mst.number_of_edges()}")
    
    return mst
```

#### 📝 My Practice

**Exercise 17:** Find minimum spanning tree
```python
# My graph has [#] edges
# MST has [#] edges

# Edges kept in MST:
[List critical relationships]

# Interpretation:
[What does this tell me about network structure?]
```

---

### 6.2 Triangle Counting

**What it does:** Count triangles (3-node cycles) in graph.

**Use case:** Measure clustering, find tightly connected groups.

#### NetworkX
```python
def count_triangles(graph):
    """
    Count triangles in graph
    """
    triangles = nx.triangles(graph)
    
    # Total triangles
    total = sum(triangles.values()) // 3
    
    print(f"Total triangles: {total}")
    
    # Nodes in most triangles
    sorted_nodes = sorted(
        triangles.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return triangles, sorted_nodes

# Example
tri_count, top_nodes = count_triangles(G)
print("\nNodes in most triangles:")
for node, count in top_nodes[:5]:
    print(f"  {node}: {count}")
```

#### 📝 My Practice

**Exercise 18:** Count triangles in my graph
```python
# Total triangles: [#]

# Companies in most triangles:
[List with interpretation]

# What triangles tell me:
[Analysis of clustering]
```

---

### 6.3 Cycle Detection

**What it does:** Find cycles in directed graph.

**Use case:** Detect circular dependencies, feedback loops.

#### NetworkX
```python
def find_cycles(graph):
    """
    Find all cycles in directed graph
    """
    try:
        cycles = list(nx.simple_cycles(graph))
        print(f"Found {len(cycles)} cycles")
        
        for idx, cycle in enumerate(cycles[:10]):  # Show first 10
            print(f"Cycle {idx}: {' -> '.join(cycle + [cycle[0]])}")
        
        return cycles
    except:
        print("No cycles found (or graph is undirected)")
        return []
```

#### 📝 My Practice

**Exercise 19:** Find cycles in supplier relationships
```python
# Cycles I found:
[List cycles]

# What they mean:
[Interpretation - circular dependencies?]
```

---

## PART 7: Practical Exercises & Projects

### Project 1: Supply Chain Analysis

**TODO:** Analyze supply chain resilience

```python
def analyze_supply_chain(graph, company):
    """
    Comprehensive supply chain analysis
    """
    analysis = {}
    
    # 1. Direct suppliers
    suppliers = list(graph.predecessors(company))
    analysis['supplier_count'] = len(suppliers)
    
    # 2. Supplier diversity (betweenness of suppliers)
    supplier_importance = {}
    for supplier in suppliers:
        supplier_importance[supplier] = nx.betweenness_centrality(graph)[supplier]
    analysis['critical_suppliers'] = sorted(
        supplier_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 3. Alternative paths
    # If critical supplier fails, are there alternatives?
    
    # 4. Supply chain depth
    # How many hops to raw materials?
    
    return analysis

# My implementation:
[Complete this function for my graph]
```

**My results:**
```
Company: [Ticker]
Direct suppliers: [#]
Critical suppliers: [List]
Risk level: [High/Medium/Low]
Recommendations: [My insights]
```

---

### Project 2: Competitive Landscape

**TODO:** Map competitive relationships

```python
def analyze_competition(graph, company):
    """
    Analyze competitive positioning
    """
    # 1. Direct competitors
    
    # 2. Indirect competitors (shared customers)
    
    # 3. Competitive strength (PageRank, centrality)
    
    # 4. Market positioning
    
    pass

# My analysis for 3 companies:
[Results]
```

---

### Project 3: Network Vulnerability

**TODO:** Find single points of failure

```python
def find_vulnerabilities(graph):
    """
    Find nodes whose removal disconnects graph
    """
    # Articulation points
    articulation_points = list(nx.articulation_points(graph))
    
    print(f"Found {len(articulation_points)} critical nodes")
    
    # For each, see what happens if removed
    for node in articulation_points:
        graph_copy = graph.copy()
        graph_copy.remove_node(node)
        components = list(nx.connected_components(graph_copy))
        print(f"Removing {node}: {len(components)} components")
    
    return articulation_points

# My critical nodes:
[List and explain why they're critical]
```

---

### Project 4: Investment Clustering

**TODO:** Group companies by investment characteristics

```python
def cluster_investment_opportunities(graph):
    """
    Use multiple algorithms to cluster companies
    """
    # 1. Community detection (structure)
    communities = detect_communities_louvain(graph)
    
    # 2. PageRank (importance)
    pagerank = calculate_pagerank(graph)
    
    # 3. Combine with features (sector, market cap, etc.)
    
    # 4. Create investment groups
    
    pass

# My investment clusters:
# Cluster 1: [Companies] - Strategy: [Buy/Hold/Sell]
# Cluster 2: [Companies] - Strategy: [Buy/Hold/Sell]
```

---

## Algorithm Comparison Table

**I need to fill this out as I learn:**

| Algorithm | What It Finds | Time Complexity | When to Use | My Notes |
|-----------|---------------|-----------------|-------------|----------|
| BFS | Shortest path | O(V+E) | Unweighted graphs | |
| Dijkstra | Weighted shortest path | O(E log V) | Weighted graphs | |
| PageRank | Node importance | O(V+E) * k | Influence ranking | |
| Betweenness | Bridge nodes | O(V*E) | Find connectors | |
| Louvain | Communities | O(V log V) | Clustering | |
| Jaccard | Similarity | O(V) per pair | Link prediction | |

---

## Performance Benchmarks

**I need to track this on my graph:**

| Algorithm | My Graph Size | Execution Time | Memory Used | Notes |
|-----------|---------------|----------------|-------------|-------|
| PageRank | [# nodes/edges] | | | |
| Louvain | | | | |
| Betweenness | | | | |

---

## My Algorithm Toolkit

**Essential queries I use frequently:**

```cypher
// 1. [My most used query]
[Query here]

// 2. [Second most used]
[Query here]

// 3. [Third most used]
[Query here]
```

---

## Common Mistakes I Made

### Mistake 1: [What I did wrong]
**Problem:** [What happened]
**Solution:** [How I fixed it]

### Mistake 2: [What I did wrong]
**Solution:** [How I fixed it]

---

## Next Steps

- [ ] Complete [06-temporal-graphs.md](06-temporal-graphs.md)
- [ ] Implement all algorithms on my graph
- [ ] Complete all TODO exercises
- [ ] Create my own algorithm combinations

---

## Resources I Used

- [ ] NetworkX Documentation: https://networkx.org/documentation/stable/reference/algorithms/
- [ ] Neo4j Graph Data Science: https://neo4j.com/docs/graph-data-science/
- [ ] Book: [Which book]
- [ ] Video: [Which video]

---

## My Notes & Discoveries

[Algorithms that surprised me, patterns I found, insights gained]

---

**Status:** [Date] - [What I learned today]
