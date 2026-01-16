# 01 - Graph Basics

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

Understanding the fundamental concepts of graph theory that form the foundation of this project.

---

## Core Concepts

### 1. Nodes (Vertices)

**What they are:** The "things" in my system - companies, people, products, events.

**Example:**
```
Node: NVIDIA
Properties:
  - ticker: "NVDA"
  - name: "NVIDIA Corporation"
  - sector: "Semiconductor"
  - market_cap: 1200000000000
```

**My understanding:**
[Write notes here after learning]

---

### 2. Edges (Relationships)

**What they are:** The connections between nodes. They represent how things relate to each other.

**Example:**
```
NVIDIA --[SUPPLIES_TO]--> Tesla
Properties:
  - deal_value: 500000000
  - confidence: 0.87
  - established_date: "2024-01-16"
```

**My understanding:**
[Write notes here]

---

### 3. Directed vs Undirected

**Directed:** Relationship has direction (A supplies to B, but B doesn't supply to A)
```
NVIDIA --> Tesla  (supplier relationship)
```

**Undirected:** Relationship goes both ways (A competes with B means B competes with A)
```
NVIDIA <--> AMD  (competition)
```

**My examples:**
[Create my own examples here]

---

### 4. Properties

**What they are:** Data attached to nodes and edges.

**For Nodes:**
- ticker symbol
- company name
- market cap
- sector

**For Edges:**
- weight/strength
- confidence score
- date established
- source of information

**Why properties matter:**
[My thoughts on why this is important]

---

## Practical Exercise

### Exercise 1: Design on Paper

Draw a simple graph with 5 companies I know:

```
[Draw here or describe]

Companies:
1. 
2. 
3. 
4. 
5. 

Relationships:
1. 
2. 
3. 
```

### Exercise 2: Real-World Mapping

Pick 3 companies from my domain and map their relationships:

**Company 1:** [Name]
**Company 2:** [Name]  
**Company 3:** [Name]

**Relationships I know:**
- [Company] --[relationship type]--> [Company] because [reason]
- [Company] --[relationship type]--> [Company] because [reason]

---

## Key Questions to Answer

Before moving on, I should be able to answer:

- [ ] What's the difference between a node and an edge?
- [ ] When should I use a directed vs undirected relationship?
- [ ] What information should be a property vs a separate node?
- [ ] Can two nodes have multiple relationships? (Answer: Yes!)

**My answers:**
[Write my understanding here]

---

## Common Mistakes to Avoid

### ❌ Storing lists as properties
```
company.competitors = ['AMD', 'INTC', 'QCOM']
```

### ✅ Creating explicit relationships
```
NVIDIA --[COMPETES_WITH {intensity: 0.89}]--> AMD
NVIDIA --[COMPETES_WITH {intensity: 0.72}]--> INTC
NVIDIA --[COMPETES_WITH {intensity: 0.54}]--> QCOM
```

**Why this matters:**
[My notes on why]

---

## Resources I Used

- [ ] Video: [Link and notes]
- [ ] Article: [Link and key takeaways]
- [ ] Tutorial: [Link and what I built]

---

## My Implementation Ideas

Ideas for how I'll use graphs in my project:

1. [Idea 1]
2. [Idea 2]
3. [Idea 3]

---

## Next Steps

After completing this module:
- [ ] Move to [02-graph-database.md](02-graph-database.md)
- [ ] Review notes if anything unclear
- [ ] Try explaining concepts to someone else

---

## My Notes & Insights

[Space for my personal observations, aha moments, and questions]

---

**Status Update:** [Date] - [What I learned today]
