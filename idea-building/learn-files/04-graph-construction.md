# 04 - Graph Construction

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Completed
- [ ] Reviewed

**Time Estimate:** 1-2 weeks  
**Started:** [Date]  
**Completed:** [Date]

---

## What I Need to Learn

Building systems that automatically create and update graph relationships based on rules and incoming data.

---

## Prerequisites

- [ ] Completed modules 01-03
- [ ] Have working graph database
- [ ] Understand my schema

---

## Core Concepts

### 1. Rule-Based Edge Creation

**When do I create a relationship?**

```python
def should_create_edge(entity_a, entity_b, relationship_type, confidence):
    """
    My decision logic
    """
    # Rule 1: Confidence threshold
    if confidence < 0.7:
        return False
    
    # Rule 2: Check if already exists
    existing = self.check_existing_edge(entity_a, entity_b, relationship_type)
    if existing:
        return False  # Will update instead
    
    # Rule 3: Business logic
    if entity_a == entity_b:
        return False  # No self-loops
    
    # Rule 4: Relationship makes sense
    if relationship_type == 'SUPPLIES_TO' and entity_a.type == 'Person':
        return False  # People don't supply to companies
    
    return True
```

**My rules I'm implementing:**
- [ ] Rule 1: [My rule]
- [ ] Rule 2: [My rule]
- [ ] Rule 3: [My rule]

---

### 2. Weight Calculation

**How I calculate edge weights:**

```python
def calculate_edge_weight(signals):
    """
    Combine multiple signals into one weight
    """
    # Option A: Weighted average
    weight = (
        signals['nlp_confidence'] * 0.5 +
        signals['sentiment'] * 0.3 +
        signals['source_credibility'] * 0.2
    )
    
    # Option B: Minimum (conservative)
    weight = min(
        signals['nlp_confidence'],
        signals['sentiment'],
        signals['source_credibility']
    )
    
    # Option C: Custom formula
    weight = (signals['nlp_confidence'] ** 0.5) * signals['sentiment']
    
    return min(weight, 1.0)  # Cap at 1.0
```

**My approach:**
```python
# My weight calculation
[My actual implementation]
```

---

### 3. Update vs Create Logic

```python
def process_new_relationship(source, target, rel_type, new_weight):
    """
    Decide: create new or update existing?
    """
    existing = self.find_edge(source, target, rel_type)
    
    if existing:
        # Update strategy
        if new_weight > existing.weight + 0.1:
            # New evidence is much stronger
            existing.weight = new_weight
        else:
            # Average with existing
            existing.weight = (existing.weight + new_weight) / 2
        
        existing.last_updated = now()
        existing.update_count += 1
        
    else:
        # Create new edge
        self.create_edge(source, target, rel_type, new_weight)
```

**My update strategy:**
[How I handle updates]

---

## My Implementation

### Step 1: Basic Edge Creator

```python
from neo4j import GraphDatabase

class GraphConstructor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def create_relationship(self, from_ticker, to_ticker, rel_type, properties):
        """
        Create a relationship between two companies
        """
        with self.driver.session() as session:
            query = """
            MATCH (a:Company {ticker: $from_ticker})
            MATCH (b:Company {ticker: $to_ticker})
            MERGE (a)-[r:%s]->(b)
            SET r += $properties
            SET r.last_updated = timestamp()
            RETURN r
            """ % rel_type
            
            result = session.run(
                query,
                from_ticker=from_ticker,
                to_ticker=to_ticker,
                properties=properties
            )
            return result.single()
```

**My implementation:**
```python
# My actual code
[My graph constructor class]
```

---

### Step 2: Rule Engine

```python
class RuleEngine:
    def __init__(self, graph_constructor):
        self.graph = graph_constructor
        self.rules = self.load_rules()
    
    def process_extracted_relationship(self, entity_a, entity_b, rel_info):
        """
        Apply rules to decide if/how to create relationship
        """
        # Check confidence
        if rel_info['confidence'] < 0.7:
            print(f"Skipped: Low confidence {rel_info['confidence']}")
            return None
        
        # Check relationship makes sense
        if not self.validate_relationship(entity_a, entity_b, rel_info['type']):
            print(f"Skipped: Invalid relationship type")
            return None
        
        # Calculate weight
        weight = self.calculate_weight(rel_info)
        
        # Check if exists
        existing = self.graph.find_relationship(
            entity_a, entity_b, rel_info['type']
        )
        
        if existing:
            return self.update_relationship(existing, weight, rel_info)
        else:
            return self.create_relationship(entity_a, entity_b, rel_info, weight)
    
    def validate_relationship(self, entity_a, entity_b, rel_type):
        """
        Business logic validation
        """
        # Rule: No self-loops
        if entity_a == entity_b:
            return False
        
        # Rule: Type compatibility
        valid_types = {
            'SUPPLIES_TO': ('Company', 'Company'),
            'COMPETES_WITH': ('Company', 'Company'),
            'HAS_EXECUTIVE': ('Person', 'Company'),
        }
        
        expected = valid_types.get(rel_type)
        if expected:
            return (entity_a.type, entity_b.type) == expected
        
        return True
```

**My rule engine:**
```python
# My implementation
[My code]
```

---

### Step 3: Batch Processing

```python
def process_batch(self, relationships_list):
    """
    Process multiple relationships efficiently
    """
    created = 0
    updated = 0
    skipped = 0
    
    for rel in relationships_list:
        try:
            result = self.process_extracted_relationship(
                rel['source'],
                rel['target'],
                rel['info']
            )
            
            if result == 'created':
                created += 1
            elif result == 'updated':
                updated += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"Error processing {rel}: {e}")
            skipped += 1
    
    return {
        'created': created,
        'updated': updated,
        'skipped': skipped
    }
```

**My batch processor:**
[My implementation]

---

## Conflict Resolution

### Scenario 1: Competing Relationships

**Problem:** News says "partnership" but graph says "competitor"

**My strategy:**
```python
def handle_conflicting_relationships(existing_edge, new_relationship):
    """
    What I do when relationships conflict
    """
    # Strategy A: Keep both
    if existing_edge.type == 'COMPETES_WITH' and new_relationship == 'SUPPLIES_TO':
        # They can compete AND partner
        self.create_edge(new_relationship)
        # Reduce competition intensity
        existing_edge.weight *= 0.7
    
    # Strategy B: Replace if much higher confidence
    elif new_confidence > existing_edge.confidence + 0.3:
        self.delete_edge(existing_edge)
        self.create_edge(new_relationship)
    
    # Strategy C: Merge/average
    else:
        existing_edge.weight = (existing_edge.weight + new_weight) / 2
```

**My approach:**
[How I handle conflicts]

---

### Scenario 2: Confidence Updates

**When new evidence arrives:**

```python
def update_confidence(existing_edge, new_evidence):
    """
    Update edge confidence with new information
    """
    old_conf = existing_edge.confidence
    new_conf = new_evidence.confidence
    
    # Bayesian update (simplified)
    updated_conf = (old_conf * existing_edge.evidence_count + new_conf) / (existing_edge.evidence_count + 1)
    
    existing_edge.confidence = updated_conf
    existing_edge.evidence_count += 1
    existing_edge.last_updated = now()
```

**My confidence tracking:**
[My implementation]

---

## Data Pipeline

### My complete flow:

```
1. Raw Data (news article, filing, etc.)
   ↓
2. NLP Extraction
   → entities: ['NVDA', 'TSLA']
   → relationship: 'SUPPLIES_TO'
   → confidence: 0.92
   ↓
3. Rule Engine
   → validate relationship
   → calculate weight
   → check existing
   ↓
4. Graph Update
   → create or update edge
   → update timestamps
   ↓
5. Post-Processing
   → recalculate metrics
   → trigger alerts
   → log changes
```

**My pipeline code:**
```python
[My data pipeline implementation]
```

---

## Testing My Rules

### Test Case 1: High Confidence
```python
# Input
entity_a = 'NVDA'
entity_b = 'TSLA'
rel_info = {
    'type': 'SUPPLIES_TO',
    'confidence': 0.92,
    'source': 'Reuters'
}

# Expected output
# → Create SUPPLIES_TO edge with weight ~0.92

# Actual result
[My test result]
```

### Test Case 2: Low Confidence
```python
# Input
rel_info = {'confidence': 0.45}

# Expected output
# → Skip, below threshold

# Actual result
[My test result]
```

### Test Case 3: Duplicate
```python
# Input
# Same relationship already exists

# Expected output
# → Update existing edge, increase confidence

# Actual result
[My test result]
```

**My test suite:**
- [ ] Test high confidence creation
- [ ] Test low confidence rejection
- [ ] Test duplicate handling
- [ ] Test conflict resolution
- [ ] Test weight calculation
- [ ] Test batch processing

---

## Monitoring & Logging

### What I track:

```python
class GraphConstructionMetrics:
    def __init__(self):
        self.edges_created = 0
        self.edges_updated = 0
        self.edges_skipped = 0
        self.errors = []
        self.processing_time = []
    
    def log_creation(self, edge_info, duration):
        self.edges_created += 1
        self.processing_time.append(duration)
        
        # Log to file
        logger.info(f"Created: {edge_info}")
    
    def report(self):
        return {
            'total_created': self.edges_created,
            'total_updated': self.edges_updated,
            'total_skipped': self.edges_skipped,
            'avg_processing_time': np.mean(self.processing_time),
            'error_rate': len(self.errors) / (self.edges_created + self.edges_updated)
        }
```

**My metrics:**
[What I'm tracking]

---

## Common Issues I Faced

### Issue 1: [Problem]
**Symptoms:** [What happened]
**Root cause:** [Why it happened]
**Solution:** [How I fixed it]

### Issue 2: [Problem]
**Solution:** [How I fixed it]

---

## Performance Optimization

### Batch Operations
```python
# Instead of one-by-one
for rel in relationships:
    create_edge(rel)  # Slow!

# Do batch create
batch_create_edges(relationships)  # Fast!
```

### Using MERGE vs CREATE
```cypher
// MERGE: Check if exists, create if not
MERGE (a:Company {ticker: 'NVDA'})-[r:SUPPLIES_TO]->(b:Company {ticker: 'TSLA'})
SET r.weight = 0.87

// More efficient than separate MATCH + CREATE
```

**My optimizations:**
[What I did to improve performance]

---

## My Configuration

```python
# config.py

# Thresholds
MIN_CONFIDENCE = 0.70
MIN_WEIGHT = 0.50
MAX_WEIGHT = 1.00

# Update strategies
UPDATE_STRATEGY = 'weighted_average'  # or 'replace', 'maximum'

# Weights for different signal types
SIGNAL_WEIGHTS = {
    'nlp_confidence': 0.50,
    'sentiment': 0.30,
    'source_credibility': 0.20
}

# Source credibility scores
SOURCE_CREDIBILITY = {
    'Reuters': 0.95,
    'Bloomberg': 0.95,
    'Twitter': 0.60,
    'Reddit': 0.40
}
```

**My config:**
[My actual configuration]

---

## Next Steps

- [ ] Complete [05-graph-algorithms.md](05-graph-algorithms.md)
- [ ] Test with real data
- [ ] Optimize batch processing
- [ ] Add error handling

---

## My Notes

[Implementation challenges, decisions made, lessons learned]

---

**Status:** [Date] - [Progress]
