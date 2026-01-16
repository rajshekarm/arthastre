# 02 - Graph Database

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Completed
- [ ] Reviewed

**Time Estimate:** 2-3 weeks  
**Started:** [Date]  
**Completed:** [Date]

---

## What I Need to Learn

Setting up Neo4j and learning Cypher query language to interact with my graph database.

---

## Prerequisites

Before starting:
- [ ] Completed [01-graph-basics.md](01-graph-basics.md)
- [ ] Understand nodes, edges, properties
- [ ] Have examples I want to implement

---

## Setup Checklist

### Week 1: Installation & First Database

#### Day 1-2: Environment Setup
- [ ] Download Neo4j Desktop from https://neo4j.com/download/
- [ ] Install Neo4j Desktop
- [ ] Create my first database
- [ ] Start the database
- [ ] Open Neo4j Browser

**Installation notes:**
[Problems I encountered and how I solved them]

#### Day 3-4: Neo4j Browser Tour
- [ ] Understand the interface
- [ ] Find the query editor
- [ ] Run the built-in tutorial (`:play intro`)
- [ ] Explore the visualization

**Interface notes:**
[How the interface works, shortcuts I learned]

---

## Cypher Query Language

### Basic Syntax I Need to Master

#### 1. Creating Nodes

```cypher
// Create a single company
CREATE (n:Company {
  ticker: 'NVDA',
  name: 'NVIDIA Corporation',
  sector: 'Semiconductor'
})

// Create multiple nodes
CREATE 
  (tesla:Company {ticker: 'TSLA', name: 'Tesla Inc.'}),
  (amd:Company {ticker: 'AMD', name: 'Advanced Micro Devices'})
```

**My practice:**
```cypher
// I created these nodes:
[My actual queries here]
```

#### 2. Finding Nodes

```cypher
// Find all companies
MATCH (c:Company)
RETURN c

// Find specific company
MATCH (c:Company {ticker: 'NVDA'})
RETURN c

// Find with conditions
MATCH (c:Company)
WHERE c.market_cap > 1000000000000
RETURN c.name, c.market_cap
```

**My practice queries:**
```cypher
[Queries I wrote to find my data]
```

#### 3. Creating Relationships

```cypher
// Create relationship between existing nodes
MATCH (nvidia:Company {ticker: 'NVDA'})
MATCH (tesla:Company {ticker: 'TSLA'})
CREATE (nvidia)-[:SUPPLIES_TO {
  weight: 0.87,
  deal_value: 500000000
}]->(tesla)

// Create nodes and relationship in one go
CREATE (a:Company {ticker: 'AAPL'})-[:COMPETES_WITH {intensity: 0.6}]->(b:Company {ticker: 'MSFT'})
```

**My practice:**
```cypher
[My relationship creation queries]
```

#### 4. Finding Relationships

```cypher
// Find all relationships
MATCH (a)-[r]->(b)
RETURN a, r, b
LIMIT 25

// Find specific relationship type
MATCH (a)-[r:SUPPLIES_TO]->(b)
RETURN a.name AS supplier, b.name AS customer, r.deal_value

// Find paths
MATCH path = (a:Company {ticker: 'NVDA'})-[*1..2]-(b)
RETURN path
```

**Queries I ran:**
```cypher
[My exploration queries]
```

#### 5. Updating Data

```cypher
// Update node properties
MATCH (c:Company {ticker: 'NVDA'})
SET c.market_cap = 1200000000000,
    c.last_updated = timestamp()

// Update relationship properties
MATCH (a)-[r:SUPPLIES_TO]->(b)
WHERE a.ticker = 'NVDA' AND b.ticker = 'TSLA'
SET r.weight = 0.90

// Add new property
MATCH (c:Company)
SET c.country = 'USA'
```

**My updates:**
```cypher
[Properties I updated in my graph]
```

#### 6. Deleting Data

```cypher
// Delete a relationship
MATCH (a)-[r:SUPPLIES_TO]->(b)
WHERE a.ticker = 'NVDA'
DELETE r

// Delete a node (must delete relationships first!)
MATCH (c:Company {ticker: 'OLD_COMPANY'})
DETACH DELETE c  // DETACH deletes relationships too

// Delete all data (CAREFUL!)
MATCH (n)
DETACH DELETE n
```

**When I needed to delete:**
[Scenarios where I deleted data]

---

## My First Real Database

### Week 2: Building My Dataset

#### Step 1: Import 20 Companies

```cypher
// Create companies one by one or from CSV
CREATE 
  (c1:Company {ticker: 'NVDA', name: 'NVIDIA Corporation', sector: 'Semiconductor'}),
  (c2:Company {ticker: 'TSLA', name: 'Tesla Inc.', sector: 'Automotive'}),
  (c3:Company {ticker: 'AMD', name: 'Advanced Micro Devices', sector: 'Semiconductor'}),
  // ... add 17 more
```

**My 20 companies:**
1. [Ticker] - [Name]
2. [Ticker] - [Name]
3. [Ticker] - [Name]
... [continue list]

**Query I used:**
```cypher
[My actual creation query]
```

#### Step 2: Create Relationships

```cypher
// Create supplier relationships
MATCH (nvda:Company {ticker: 'NVDA'}),
      (tsla:Company {ticker: 'TSLA'})
CREATE (nvda)-[:SUPPLIES_TO {weight: 0.87, product: 'AI chips'}]->(tsla)

// Create competition relationships
MATCH (nvda:Company {ticker: 'NVDA'}),
      (amd:Company {ticker: 'AMD'})
CREATE (nvda)-[:COMPETES_WITH {intensity: 0.89, market: 'GPU'}]->(amd)
```

**My relationships (at least 50):**
- [ ] Created 10 SUPPLIES_TO relationships
- [ ] Created 15 COMPETES_WITH relationships
- [ ] Created 10 PARTNERS_WITH relationships
- [ ] Created 15 IN_SECTOR relationships

**Queries I used:**
```cypher
[My relationship creation queries]
```

---

## Useful Query Patterns I Learned

### Pattern 1: Find Neighbors
```cypher
// Who does NVIDIA work with?
MATCH (nvda:Company {ticker: 'NVDA'})-[r]-(other)
RETURN other.name, type(r), r
```

### Pattern 2: Count Relationships
```cypher
// How many suppliers does each company have?
MATCH (supplier)-[:SUPPLIES_TO]->(company)
RETURN company.name, count(supplier) AS supplier_count
ORDER BY supplier_count DESC
```

### Pattern 3: Find Paths
```cypher
// How is NVIDIA connected to Apple?
MATCH path = shortestPath(
  (nvda:Company {ticker: 'NVDA'})-[*]-(aapl:Company {ticker: 'AAPL'})
)
RETURN path
```

**Patterns I used most:**
```cypher
[My frequently used queries]
```

---

## Visualization

### What I Learned About Neo4j Browser

- Nodes appear as circles
- Relationships are arrows
- Colors represent node types
- Can expand/collapse nodes
- Can style based on properties

**My visualization settings:**
[How I configured the visualization]

---

## Performance & Indexing

### When My Queries Got Slow

```cypher
// Create index on ticker for faster lookups
CREATE INDEX company_ticker FOR (c:Company) ON (c.ticker)

// Create index on name
CREATE INDEX company_name FOR (c:Company) ON (c.name)

// Check existing indexes
SHOW INDEXES
```

**Indexes I created:**
```cypher
[My index creation queries]
```

---

## Common Problems I Faced

### Problem 1: [Describe problem]
**Solution:** [How I solved it]

### Problem 2: [Describe problem]
**Solution:** [How I solved it]

### Problem 3: [Describe problem]
**Solution:** [How I solved it]

---

## Data Import Methods

### Method 1: Manual Creation
Good for: Small datasets, learning
```cypher
CREATE (n:Company {ticker: 'NVDA', name: 'NVIDIA'})
```

### Method 2: CSV Import
Good for: Larger datasets, reproducibility
```cypher
LOAD CSV WITH HEADERS FROM 'file:///companies.csv' AS row
CREATE (c:Company {
  ticker: row.ticker,
  name: row.name,
  sector: row.sector
})
```

**What I used:**
[Method I chose and why]

**My CSV structure:**
```
ticker,name,sector,market_cap
NVDA,NVIDIA Corporation,Semiconductor,1200000000000
TSLA,Tesla Inc.,Automotive,650000000000
```

---

## Week 3: Advanced Queries

### Aggregations
```cypher
// Average market cap by sector
MATCH (c:Company)
RETURN c.sector, avg(c.market_cap) AS avg_cap
ORDER BY avg_cap DESC
```

### Conditional Logic
```cypher
// Label companies by size
MATCH (c:Company)
RETURN c.name,
  CASE
    WHEN c.market_cap > 1000000000000 THEN 'Mega Cap'
    WHEN c.market_cap > 200000000000 THEN 'Large Cap'
    ELSE 'Mid Cap'
  END AS size_category
```

### Pattern Matching
```cypher
// Find companies that both compete and partner
MATCH (a:Company)-[:COMPETES_WITH]-(b:Company),
      (a)-[:PARTNERS_WITH]-(b)
RETURN a.name, b.name
```

**Advanced queries I wrote:**
```cypher
[My complex queries]
```

---

## My Database Stats

After completing this module:
- Total nodes: [Number]
- Total relationships: [Number]
- Node types: [List types]
- Relationship types: [List types]

```cypher
// Query to check my stats
MATCH (n)
RETURN count(n) AS total_nodes

MATCH ()-[r]->()
RETURN count(r) AS total_relationships

MATCH (n)
RETURN DISTINCT labels(n) AS node_types

MATCH ()-[r]->()
RETURN DISTINCT type(r) AS relationship_types
```

**My results:**
[Actual numbers from my database]

---

## Cheat Sheet I Created

### My Most Used Commands
```cypher
// [Command 1]: [What it does]

// [Command 2]: [What it does]

// [Command 3]: [What it does]
```

---

## Practice Exercises Completed

- [ ] Exercise 1: Create 20 company nodes
- [ ] Exercise 2: Create 50 relationships
- [ ] Exercise 3: Find all competitors of NVIDIA
- [ ] Exercise 4: Find shortest path between two companies
- [ ] Exercise 5: Count relationships per company
- [ ] Exercise 6: Update properties based on condition
- [ ] Exercise 7: Delete and recreate a relationship

**Notes on exercises:**
[What I learned from each]

---

## Resources I Used

- [ ] Neo4j Getting Started Guide: [Link]
- [ ] Cypher Manual: https://neo4j.com/docs/cypher-manual/
- [ ] YouTube Tutorial: [Which one and notes]
- [ ] GraphAcademy Course: [Which course]

**Most helpful resource:**
[What helped me most and why]

---

## Next Steps

After completing this module:
- [ ] Move to [03-graph-modeling.md](03-graph-modeling.md)
- [ ] Backup my database
- [ ] Export my schema for reference
- [ ] Practice Cypher daily for 1 week

---

## My Notes & Insights

[Personal observations, breakthrough moments, things that clicked]

---

**Status Update:** [Date] - [What I accomplished today]
