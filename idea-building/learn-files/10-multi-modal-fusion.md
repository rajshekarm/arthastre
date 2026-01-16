# 10 - Multi-Modal Fusion

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

Combining three different data types (graph relationships, text sentiment, price movements) into a unified system that provides better insights than any single source alone.

---

## The Three Modalities

### 1. Graph Features (Structured)
```python
graph_features = {
    'pagerank': 0.0234,
    'degree': 47,
    'centrality': 0.89,
    'clustering_coef': 0.67,
    'community': 'AI_Infrastructure',
    'embedding': [0.1, 0.2, ...]  # From GNN
}
```

### 2. Text Features (Unstructured)
```python
text_features = {
    'sentiment': 0.85,
    'recent_mentions': 1247,
    'sentiment_trend': 'improving',
    'key_themes': ['AI leadership', 'strong demand'],
    'text_embedding': [0.3, -0.1, ...]  # From BERT
}
```

### 3. Price Features (Time-Series)
```python
price_features = {
    'return_1d': 0.012,
    'volatility_30d': 0.23,
    'volume_ratio': 2.3,
    'rsi': 68,
    'momentum': 'STRONG',
    'above_ma50': True
}
```

---

## The Core Problem

**Each modality sees a different part of reality:**

- **Graph:** NVIDIA is central in AI infrastructure network
- **Text:** News about NVIDIA is overwhelmingly positive  
- **Price:** NVIDIA stock is up with high volume

**Question:** Are all three telling the same story?

---

## Cross-Modal Validation

### Step 1: Extract Signals

```python
def extract_signals(company_ticker):
    """
    Get signal from each modality
    """
    # Graph signal
    graph_signal = {
        'direction': 'BULLISH' if pagerank_increasing else 'BEARISH',
        'strength': compute_graph_strength(company_ticker),
        'confidence': 0.85
    }
    
    # Text signal
    text_signal = {
        'direction': 'BULLISH' if sentiment > 0.6 else 'BEARISH',
        'strength': abs(sentiment - 0.5) * 2,  # 0 to 1 scale
        'confidence': 0.90
    }
    
    # Price signal
    price_signal = {
        'direction': 'BULLISH' if return_1d > 0 else 'BEARISH',
        'strength': abs(return_1d) * 10,  # Normalize
        'confidence': 0.75  # Markets can be noisy
    }
    
    return graph_signal, text_signal, price_signal
```

**My implementation:**
```python
[My signal extraction code]
```

---

### Step 2: Check Agreement

```python
def check_agreement(signals):
    """
    Do all three modalities agree?
    """
    directions = [s['direction'] for s in signals]
    
    # All pointing same direction?
    if len(set(directions)) == 1:
        # Perfect agreement
        avg_confidence = np.mean([s['confidence'] for s in signals])
        avg_strength = np.mean([s['strength'] for s in signals])
        
        return {
            'agreement': True,
            'direction': directions[0],
            'confidence': avg_confidence * 1.2,  # Boost for agreement
            'strength': avg_strength
        }
    
    # Majority agreement?
    elif directions.count('BULLISH') == 2 or directions.count('BEARISH') == 2:
        majority = 'BULLISH' if directions.count('BULLISH') == 2 else 'BEARISH'
        
        return {
            'agreement': 'PARTIAL',
            'direction': majority,
            'confidence': avg_confidence * 0.9,  # Slight penalty
            'investigate': True  # Flag for review
        }
    
    # Complete disagreement
    else:
        return {
            'agreement': False,
            'confidence': 0.3,  # Very low confidence
            'action': 'HOLD',  # Don't act on conflicting signals
            'investigate': True
        }
```

**My validation logic:**
```python
[My implementation]
```

---

## Feature Engineering

### Combining Features from Different Modalities

```python
from sklearn.preprocessing import StandardScaler

class MultiModalFeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def prepare_features(self, company_ticker):
        """
        Prepare features from all modalities
        """
        # Get raw features
        graph_feat = self.get_graph_features(company_ticker)
        text_feat = self.get_text_features(company_ticker)
        price_feat = self.get_price_features(company_ticker)
        
        # Normalize to same scale
        graph_norm = self.normalize(graph_feat)
        text_norm = self.normalize(text_feat)
        price_norm = self.normalize(price_feat)
        
        # Option 1: Concatenate
        concat_features = np.concatenate([
            graph_norm,
            text_norm,
            price_norm
        ])
        
        # Option 2: Weighted combination
        weighted_features = (
            graph_norm * 0.4 +
            text_norm * 0.3 +
            price_norm * 0.3
        )
        
        # Option 3: Keep separate for attention mechanism
        separate_features = {
            'graph': graph_norm,
            'text': text_norm,
            'price': price_norm
        }
        
        return concat_features  # or weighted, or separate
```

**My approach:**
```python
[Which method I'm using]
```

---

## Attention Mechanism

### Learning Which Modality to Trust

```python
import torch
import torch.nn as nn

class MultiModalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Learned query vector
        self.query = nn.Parameter(torch.randn(feature_dim))
        
    def forward(self, graph_features, text_features, price_features):
        """
        Compute attention weights for each modality
        """
        # Stack features
        features = torch.stack([
            graph_features,
            text_features,
            price_features
        ], dim=1)  # Shape: [batch, 3, feature_dim]
        
        # Compute attention scores
        scores = torch.matmul(features, self.query)  # [batch, 3]
        
        # Softmax to get weights
        weights = torch.softmax(scores, dim=1)  # [batch, 3]
        
        # Weighted combination
        combined = torch.sum(
            features * weights.unsqueeze(-1),
            dim=1
        )
        
        return combined, weights

# Example
attention = MultiModalAttention(feature_dim=128)
combined_features, weights = attention(graph_feat, text_feat, price_feat)

# weights might be: [0.5, 0.3, 0.2]
# → Graph features are most important (50%)
```

**My attention implementation:**
```python
[My code]
```

---

## Real Example: Investment Decision

### Question: "Should I buy NVIDIA?"

```python
def make_investment_decision(ticker):
    """
    Multi-modal analysis for investment decision
    """
    # Step 1: Gather data from all modalities
    graph_data = get_graph_analysis(ticker)
    text_data = get_text_analysis(ticker)
    price_data = get_price_analysis(ticker)
    
    # Step 2: Extract signals
    graph_signal = {
        'signal': 'BUY',
        'reason': 'Central node in AI infrastructure',
        'confidence': 0.85
    }
    
    text_signal = {
        'signal': 'BUY',
        'reason': 'Sentiment: 0.91, mentions up 300%',
        'confidence': 0.90
    }
    
    price_signal = {
        'signal': 'BUY',
        'reason': 'Strong momentum, volume confirms',
        'confidence': 0.75
    }
    
    # Step 3: Check agreement
    if all_agree([graph_signal, text_signal, price_signal]):
        # All three say BUY - high confidence
        final_decision = {
            'action': 'STRONG BUY',
            'confidence': 0.92,  # Average with boost
            'evidence': [
                '✅ Graph: Central in AI network',
                '✅ Text: Very positive sentiment',
                '✅ Price: Strong upward momentum'
            ]
        }
    
    elif majority_agree([graph_signal, text_signal, price_signal]):
        # 2 out of 3 - moderate confidence
        final_decision = {
            'action': 'BUY',
            'confidence': 0.75,
            'caution': 'One modality disagrees - investigate'
        }
    
    else:
        # Conflicting signals - be cautious
        final_decision = {
            'action': 'HOLD',
            'confidence': 0.30,
            'reason': 'Conflicting signals across modalities'
        }
    
    return final_decision
```

**My decision logic:**
```python
[My implementation]
```

---

## Ensemble Methods

### Combining Predictions from Different Models

```python
class MultiModalEnsemble:
    def __init__(self):
        # Three separate models, one per modality
        self.graph_model = GraphBasedClassifier()
        self.text_model = TextBasedClassifier()
        self.price_model = PriceBasedClassifier()
    
    def predict(self, company_ticker):
        """
        Ensemble prediction
        """
        # Get predictions from each
        graph_pred = self.graph_model.predict(ticker)  # 'BUY'
        text_pred = self.text_model.predict(ticker)    # 'BUY'
        price_pred = self.price_model.predict(ticker)  # 'HOLD'
        
        graph_conf = self.graph_model.confidence  # 0.85
        text_conf = self.text_model.confidence    # 0.90
        price_conf = self.price_model.confidence  # 0.70
        
        # Weighted voting
        votes = {
            'BUY': graph_conf + text_conf,  # 1.75
            'HOLD': price_conf,             # 0.70
            'SELL': 0
        }
        
        # Final decision: highest weighted vote
        final_action = max(votes, key=votes.get)
        final_confidence = votes[final_action] / sum(votes.values())
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'breakdown': {
                'graph': (graph_pred, graph_conf),
                'text': (text_pred, text_conf),
                'price': (price_pred, price_conf)
            }
        }
```

**My ensemble:**
```python
[My approach]
```

---

## Handling Disagreements

### When Modalities Conflict

```python
def investigate_disagreement(ticker, signals):
    """
    Figure out why modalities disagree
    """
    graph_sig, text_sig, price_sig = signals
    
    # Scenario 1: Good news but price flat
    if text_sig['direction'] == 'BULLISH' and price_sig['direction'] != 'BULLISH':
        hypothesis = "Market may have already priced in the news"
        action = "Wait for price confirmation"
    
    # Scenario 2: Price up but weak fundamentals
    elif price_sig['direction'] == 'BULLISH' and graph_sig['strength'] < 0.5:
        hypothesis = "Possible momentum trade without substance"
        action = "Be cautious, might be speculative"
    
    # Scenario 3: Strong fundamentals but negative sentiment
    elif graph_sig['strength'] > 0.8 and text_sig['sentiment'] < 0.4:
        hypothesis = "Market may be overreacting to temporary news"
        action = "Potential buying opportunity"
    
    return {
        'hypothesis': hypothesis,
        'recommended_action': action,
        'confidence': 0.60  # Lower confidence due to conflict
    }
```

**My investigation process:**
[How I handle conflicts]

---

## Creating Unified Representation

### The Final Multi-Modal Node

```python
class UnifiedCompanyRepresentation:
    def __init__(self, ticker):
        self.ticker = ticker
        
        # Graph features
        self.graph = {
            'pagerank': 0.0234,
            'degree': 47,
            'embedding': [...]
        }
        
        # Text features
        self.text = {
            'sentiment': 0.85,
            'themes': ['AI', 'growth'],
            'embedding': [...]
        }
        
        # Price features
        self.price = {
            'return_1d': 0.012,
            'momentum': 'STRONG',
            'rsi': 68
        }
        
        # Fused insights
        self.fused = {
            'overall_signal': 'STRONG_BUY',
            'confidence': 0.92,
            'multi_modal_agreement': True,
            'risk_score': 0.15
        }
    
    def to_vector(self):
        """
        Convert everything to a single vector
        """
        return np.concatenate([
            self.graph['embedding'],
            self.text['embedding'],
            normalize(self.price_features)
        ])
```

**My unified representation:**
```python
[My implementation]
```

---

## Storing Fusion Results

### Back to Neo4j

```cypher
// Store multi-modal insights
MATCH (c:Company {ticker: 'NVDA'})
SET c.fusion_signal = 'STRONG_BUY',
    c.fusion_confidence = 0.92,
    c.graph_strength = 0.87,
    c.text_sentiment = 0.85,
    c.price_momentum = 'STRONG',
    c.modalities_agree = true,
    c.last_fusion_update = timestamp()
```

---

## Real-Time Updates

### When New Data Arrives

```python
def update_fusion_on_new_data(ticker, data_type, new_data):
    """
    Incrementally update fusion when one modality changes
    """
    current_fusion = get_current_fusion(ticker)
    
    if data_type == 'news':
        # Update text features
        new_text_signal = analyze_news(new_data)
        
        # Re-run fusion
        new_fusion = recompute_fusion(
            current_fusion['graph'],
            new_text_signal,  # Updated
            current_fusion['price']
        )
    
    elif data_type == 'price':
        # Update price features
        new_price_signal = analyze_price(new_data)
        
        new_fusion = recompute_fusion(
            current_fusion['graph'],
            current_fusion['text'],
            new_price_signal  # Updated
        )
    
    # Check if signal changed
    if new_fusion['signal'] != current_fusion['signal']:
        trigger_alert(ticker, current_fusion, new_fusion)
    
    # Store updated fusion
    store_fusion(ticker, new_fusion)
```

**My update strategy:**
[How I handle real-time updates]

---

## Testing Multi-Modal System

### Test Cases

**Test 1: Perfect Agreement**
```
Input:
- Graph: BULLISH (0.85)
- Text: BULLISH (0.90)
- Price: BULLISH (0.75)

Expected Output:
- Signal: STRONG BUY
- Confidence: ~0.92 (boosted)

Actual Result:
[My result]
```

**Test 2: Majority Agreement**
```
Input:
- Graph: BULLISH (0.85)
- Text: BULLISH (0.90)
- Price: BEARISH (0.70)

Expected Output:
- Signal: BUY (with caution)
- Confidence: ~0.75

Actual Result:
[My result]
```

**Test 3: Complete Disagreement**
```
Input:
- Graph: BULLISH
- Text: BEARISH
- Price: NEUTRAL

Expected Output:
- Signal: HOLD
- Confidence: <0.40
- Flag: Investigate

Actual Result:
[My result]
```

---

## Metrics I'm Tracking

### Fusion Performance

```python
metrics = {
    'agreement_rate': 0.67,  # 67% of time all modalities agree
    'accuracy_when_agree': 0.89,  # When they agree, 89% accurate
    'accuracy_when_disagree': 0.52,  # Much lower when conflicting
    'false_positive_rate': 0.12,
    'false_negative_rate': 0.08
}
```

**My metrics:**
[My actual numbers]

---

## Common Patterns I've Observed

### Pattern 1: Leading Indicators
```
Text sentiment rises → Graph centrality increases → Price follows
```

### Pattern 2: Price Overreaction
```
Negative news → Price drops 10% → Graph shows fundamentals strong → Price recovers
```

### Pattern 3: Delayed Graph Update
```
Major deal announced (text) → Price jumps → Graph updates next day
```

**Patterns I've found:**
[My observations]

---

## Next Steps

- [ ] Complete [11-feature-engineering.md](11-feature-engineering.md)
- [ ] Test on historical data
- [ ] Optimize fusion weights
- [ ] Deploy to production

---

## My Notes

[Key insights, challenges, breakthroughs]

---

**Status:** [Date] - [Progress]
