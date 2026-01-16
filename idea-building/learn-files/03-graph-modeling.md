# 03 - Graph Modeling

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Completed
- [ ] Reviewed

**Time Estimate:** 1 week  
**Started:** [Date]  
**Completed:** [Date]

---

## What I Need to Learn

How to design an effective graph schema for my financial intelligence system. Making smart decisions about what should be nodes, what should be relationships, and what should be properties.

---

## Prerequisites

- [ ] Completed [01-graph-basics.md](01-graph-basics.md)
- [ ] Completed [02-graph-database.md](02-graph-database.md)
- [ ] Have Neo4j running with sample data

---

## Core Design Decisions

### Decision 1: Node or Property?

**Rule of thumb I follow:**
- If I need to **query or traverse** it → Make it a **node**
- If it's just **descriptive data** → Make it a **property**

#### Examples from my project:

**✅ Good: Sector as Node**
```cypher
(NVDA:Company)-[:IN_SECTOR]->(semiconductor:Sector)
```
Why: I can query "all companies in semiconductor sector" and analyze sector relationships.

**❌ Bad: Sector as Property**
```cypher
(NVDA:Company {sector: 'Semiconductor'})
```
Why: Harder to find connections between sectors.

**My decision examples:**
| Data | Node or Property? | Why? |
|------|------------------|------|
| Company Name | Property | Just descriptive |
| Ticker Symbol | Property | Just an identifier |
| Sector | Node | Need to analyze sectors |
| CEO | Node | Need to track people |
| Market Cap | Property | Just a number |
| Product | Node | Need to analyze products |

---

### Decision 2: One Relationship or Many?

**Question:** Should NVIDIA → Tesla be one relationship with a type, or multiple?

**Option A: Multiple specific relationships**
```cypher
(NVDA)-[:SUPPLIES_CHIPS]->(TSLA)
(NVDA)-[:PARTNERS_WITH]->(TSLA)
(NVDA)-[:COMPETES_WITH]->(TSLA)
```

**Option B: One generic relationship**
```cypher
(NVDA)-[:RELATED_TO {type: 'supplier', strength: 0.87}]->(TSLA)
```

**I chose:** [A or B]

**My reasoning:**
[Why I made this choice]

---

## My Schema Design

### Node Types I'm Using

#### 1. Company
```cypher
(:Company {
  ticker: String,
  name: String,
  market_cap: Float,
  country: String,
  founded: Date,
  last_updated: Timestamp
})
```

**Why these properties:**
[My reasoning for each property]

#### 2. Person
```cypher
(:Person {
  name: String,
  role: String,
  linkedin: String
})
```

**Why I need Person nodes:**
[My reasoning]

#### 3. Sector
```cypher
(:Sector {
  name: String,
  total_market_cap: Float,
  company_count: Integer
})
```

#### 4. Product
```cypher
(:Product {
  name: String,
  category: String,
  launch_date: Date
})
```

#### 5. Event
```cypher
(:Event {
  type: String,
  timestamp: Timestamp,
  description: String,
  sentiment: Float,
  source: String
})
```

**Why Event nodes matter:**
[My reasoning for tracking events]

---

### Relationship Types I'm Using

#### 1. SUPPLIES_TO
```cypher
(supplier:Company)-[:SUPPLIES_TO {
  weight: Float,
  deal_value: Integer,
  product: String,
  confidence: Float,
  established_date: Date,
  source: String
}]->(customer:Company)
```

**Direction:** Always from supplier → customer  
**When I create this:** [My rules]

#### 2. COMPETES_WITH
```cypher
(a:Company)-[:COMPETES_WITH {
  intensity: Float,
  market: String,
  last_updated: Timestamp
}]-(b:Company)
```

**Direction:** Undirected (both ways)  
**Intensity meaning:** [How I calculate this]

#### 3. IN_SECTOR
```cypher
(company:Company)-[:IN_SECTOR {
  revenue_percentage: Float,
  primary: Boolean
}]->(sector:Sector)
```

**Note:** A company can be in multiple sectors!

#### 4. HAS_EXECUTIVE
```cypher
(person:Person)-[:HAS_EXECUTIVE {
  role: String,
  since: Date,
  ownership_pct: Float
}]->(company:Company)
```

#### 5. PRODUCES
```cypher
(company:Company)-[:PRODUCES {
  launch_date: Date,
  market_share: Float
}]->(product:Product)
```

#### 6. MENTIONED_WITH
```cypher
(a:Company)-[:MENTIONED_WITH {
  co_occurrence_count: Integer,
  sentiment_correlation: Float,
  timeframe: String
}]-(b:Company)
```

**This is auto-generated from:** [How I create these]

---

## Modeling Temporal Data

### Challenge: Relationships Change Over Time

**Problem:** NVIDIA and AMD's competition intensity changes.

**Solution I'm using:**

**Option A: Update in place with timestamp**
```cypher
(NVDA)-[:COMPETES_WITH {
  intensity: 0.89,
  last_updated: timestamp()
}]-(AMD)
```

**Option B: Create historical snapshots**
```cypher
(NVDA)-[:COMPETES_WITH_2024_01]->(AMD)
(NVDA)-[:COMPETES_WITH_2024_02]->(AMD)
```

**Option C: Event sourcing**
```cypher
(event:CompetitionUpdate {
  from: 'NVDA',
  to: 'AMD',
  old_intensity: 0.85,
  new_intensity: 0.89,
  timestamp: timestamp()
})
```

**I chose:** [My approach]

**Why:**
[My reasoning]

---

## Modeling Different Data Sources

### From News Articles
```cypher
// Event node captures the news
(event:Event {
  type: 'news',
  timestamp: timestamp(),
  source: 'Reuters',
  sentiment: 0.85,
  text_embedding: [0.1, 0.2, ...]
})

// Link to affected companies
(event)-[:AFFECTS]->(NVDA)
(event)-[:AFFECTS]->(TSLA)

// Extract relationships
(NVDA)-[:SUPPLIES_TO {source: event.id}]->(TSLA)
```

### From Price Data
```cypher
// Store as time-series properties
(NVDA:Company {
  current_price: 487.23,
  price_change_1d: 1.2,
  volume_today: 52000000,
  last_price_update: timestamp()
})
```

**Note:** I'm NOT storing full price history in graph. That goes in time-series DB.

---

## Schema Evolution Strategy

### How I handle schema changes:

1. **Adding new node type**
   - [ ] Document the new type
   - [ ] Create sample nodes
   - [ ] Add necessary relationships
   - [ ] Update my schema diagram

2. **Adding new relationship type**
   - [ ] Define what it means
   - [ ] Decide on properties
   - [ ] Determine direction
   - [ ] Create first examples

3. **Changing existing schema**
   - [ ] Don't break existing data
   - [ ] Migrate gradually
   - [ ] Keep both versions temporarily
   - [ ] Update queries

---

## My Complete Schema Diagram

```
[Draw or describe my full schema here]

Nodes:
- Company (50 nodes planned)
- Person (20 nodes)
- Sector (10 nodes)
- Product (30 nodes)
- Event (growing continuously)

Relationships:
- SUPPLIES_TO: Company → Company
- COMPETES_WITH: Company ↔ Company
- IN_SECTOR: Company → Sector
- HAS_EXECUTIVE: Person → Company
- PRODUCES: Company → Product
- AFFECTS: Event → Company/Sector/Product
- MENTIONED_WITH: Company ↔ Company
```

---

## Schema Validation Queries

### Check my schema health:

```cypher
// 1. Find orphaned nodes (no relationships)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n), count(n)

// 2. Find nodes missing required properties
MATCH (c:Company)
WHERE c.ticker IS NULL
RETURN c

// 3. Check relationship counts
MATCH (c:Company)
RETURN c.ticker, size((c)--()) AS relationship_count
ORDER BY relationship_count DESC

// 4. Find duplicate nodes
MATCH (c1:Company), (c2:Company)
WHERE c1.ticker = c2.ticker AND id(c1) < id(c2)
RETURN c1, c2
```

**Results from my database:**
[What I found]

---

## Anti-Patterns I'm Avoiding

### ❌ Anti-Pattern 1: Too Many Properties
```cypher
// BAD: Everything as properties
(company {
  ticker: 'NVDA',
  name: 'NVIDIA',
  competitor1: 'AMD',
  competitor2: 'INTC',
  supplier1: 'TSMC',
  supplier2: 'SAMSUNG'
})
```

**Why bad:** Can't query or traverse relationships

### ❌ Anti-Pattern 2: Dense Nodes
```cypher
// BAD: One node connected to everything
(hub)-[:CONNECTED_TO]->(node1)
(hub)-[:CONNECTED_TO]->(node2)
... (thousands of connections)
```

**Why bad:** Performance issues

### ❌ Anti-Pattern 3: Redundant Relationships
```cypher
// BAD: Storing same info multiple ways
(A)-[:COMPETES_WITH]->(B)
(B)-[:COMPETED_BY]->(A)  // Redundant!
```

**Why bad:** Maintenance nightmare

**Mistakes I caught myself making:**
[What I almost did wrong]

---

## Design Patterns I'm Using

### Pattern 1: Hub and Spoke
```cypher
(sector:Sector)<-[:IN_SECTOR]-(company1:Company)
(sector)<-[:IN_SECTOR]-(company2:Company)
```
**Use case:** Grouping companies by sector

### Pattern 2: Linked List (for sequences)
```cypher
(event1:Event)-[:NEXT]->(event2:Event)-[:NEXT]->(event3:Event)
```
**Use case:** Tracking event chronology

### Pattern 3: Hierarchy (for categories)
```cypher
(tech:Sector)<-[:PARENT_SECTOR]-(semiconductor:Sector)
(semiconductor)<-[:IN_SECTOR]-(NVDA:Company)
```
**Use case:** Multi-level categorization

**Patterns I'm using:**
[Which ones and where]

---

## Normalization vs Denormalization

### When I duplicate data (denormalization):

**Example:** Storing market_cap on both Company node AND Sector node
```cypher
(company:Company {market_cap: 1200000000000})
(sector:Sector {total_market_cap: 5000000000000})
```

**Why:** Performance - faster to read total without aggregation

**Trade-off:** Must keep in sync

**My approach:**
[When I denormalize and when I don't]

---

## Schema Documentation

### My Schema Changelog

**v1.0 - [Date]**
- Initial schema
- 5 node types, 7 relationship types

**v1.1 - [Date]**
- Added Event nodes
- Added MENTIONED_WITH relationships

**v1.2 - [Date]**
- [Changes I made]

---

## Example Queries Against My Schema

### Query 1: Find all suppliers of a company
```cypher
MATCH (supplier:Company)-[:SUPPLIES_TO]->(company:Company {ticker: 'TSLA'})
RETURN supplier.name, supplier.ticker
```

### Query 2: Find competitive landscape
```cypher
MATCH (company:Company {ticker: 'NVDA'})-[:COMPETES_WITH]-(competitor)
RETURN competitor.name
```

### Query 3: Find companies in same sector
```cypher
MATCH (company:Company {ticker: 'NVDA'})-[:IN_SECTOR]->(sector)<-[:IN_SECTOR]-(peer)
WHERE company <> peer
RETURN peer.name
```

### Query 4: Find indirect connections
```cypher
MATCH path = (a:Company {ticker: 'NVDA'})-[*2]-(b:Company)
RETURN path
LIMIT 10
```

**Queries I test with:**
```cypher
[My validation queries]
```

---

## Performance Considerations

### Properties I'm indexing:
```cypher
CREATE INDEX company_ticker FOR (c:Company) ON (c.ticker)
CREATE INDEX company_name FOR (c:Company) ON (c.name)
CREATE INDEX event_timestamp FOR (e:Event) ON (e.timestamp)
```

### Properties I'm NOT indexing:
- market_cap (don't query by range often)
- description (full text search later)

**My indexing strategy:**
[What I index and why]

---

## Schema Review Checklist

Before finalizing:
- [ ] Every node type has a clear purpose
- [ ] Every relationship type is well-defined
- [ ] Properties are in the right place (node vs relationship)
- [ ] No redundant data (unless intentional)
- [ ] Temporal aspects are handled
- [ ] Indexes are in place for common queries
- [ ] Schema is documented
- [ ] Example queries all work

---

## Next Steps

After completing this module:
- [ ] Move to [04-graph-construction.md](04-graph-construction.md)
- [ ] Document my final schema
- [ ] Create schema diagram
- [ ] Write schema migration scripts

---

## My Schema Design Principles

[My personal rules for schema design]

1. [Principle 1]
2. [Principle 2]
3. [Principle 3]

---

## My Notes & Insights

[Aha moments, decisions I struggled with, lessons learned]

---

**Status Update:** [Date] - [Today's progress]
