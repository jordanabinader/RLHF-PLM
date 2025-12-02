# Shared Property Model Guide

## Overview

This guide explains the **correct** architecture for personalized RLHF in protein design: one shared property model + lightweight user weights.

### The Problem with the Old Approach

The initial implementation trained separate reward models for each user:

```
❌ OLD: Separate Models Per User
PotencyMaximizer    → RewardNet₁ (100K params)
SafetyFirst         → RewardNet₂ (100K params)  
BalancedDesigner    → RewardNet₃ (100K params)
Total: N × 100K parameters
```

**Problems:**
- Redundant training: Activity, toxicity, stability are the same biology for all users
- Poor sample efficiency: Each model trained independently
- Not scalable: Adding users = training new models
- Black box: Hard to explain why a sequence is preferred

### The Correct Approach: Shared Properties + User Weights

```
✅ NEW: Shared Property Model
                       ┌─> g_act(e(x))  = p_act   ┐
Sequence x → ESM → e(x)├─> g_tox(e(x))  = p_tox   ├─> g(x) = [p_act, p_tox, p_stab, p_len]
                       ├─> g_stab(e(x)) = p_stab  │
                       └─> normalize(len(x)) = p_len┘

User weights:
PotencyMaximizer:   w₁ = [1.0,  0.0,  0.3,  0.0]  (4 params)
SafetyFirst:        w₂ = [0.5, -1.0,  0.5, -0.2]  (4 params)
BalancedDesigner:   w₃ = [0.7, -0.5,  0.6, -0.1]  (4 params)

Personalized reward: R^(u)(x) = w^(u)^T · g(x)  (simple dot product!)
```

**Benefits:**
- Property models trained once on all available data
- Adding users = defining 4 weights (not training 100K parameters)
- Interpretable: Explicit trade-offs between properties
- Fast: Matrix multiplication for rewards

## Architecture Components

### 1. Property Heads

#### Activity Head (`g_act`)
```python
# Wrapper for existing AMP classifier
activity_head = ActivityHead("amp_design/best_new_4.pth")
p_act = activity_head(esm_embedding)  # ∈ [0, 1]
```

**What it predicts:** Probability of antimicrobial activity  
**Training:** Pre-trained on AMP dataset (already done)

#### Toxicity Head (`g_tox`)
```python
# MLP trained on ToxDL2 data
toxicity_head = ToxicityHead(esm_dim=1280, domain_dim=256)
p_tox = toxicity_head(esm_embedding, domain_vector)  # ∈ [0, 1]
```

**What it predicts:** Probability of toxicity  
**Training:** Train on ToxDL2 dataset (~14K sequences)

```bash
python personalization/train_toxicity.py \
    --data-dir ToxDL2-main/data \
    --output-dir personalization/checkpoints \
    --epochs 50 \
    --device cuda
```

#### Stability Head (`g_stab`)
```python
# MLP for stability prediction
stability_head = StabilityHead(esm_dim=1280)
p_stab = stability_head(esm_embedding)  # continuous score
```

**What it predicts:** Stability score (e.g., ΔΔG)  
**Training:** Train on mega-scale stability dataset OR use EsmTherm checkpoint

```bash
# Option 1: Load pre-trained EsmTherm
python personalization/train_stability.py \
    --mode load_esmtherm \
    --esmtherm-checkpoint EsmTherm-main/output_dir/checkpoint-best

# Option 2: Train from scratch
python personalization/train_stability.py \
    --mode train \
    --train-csv data/stability_train.csv \
    --val-csv data/stability_val.csv \
    --test-csv data/stability_test.csv

# Option 3: Create placeholder (for demo)
python personalization/train_stability.py --mode placeholder
```

### 2. Unified Property Function

The `UnifiedPropertyFunction` is the single source of truth for all property predictions:

```python
from personalization import create_unified_property_function

# Create property function
property_fn = create_unified_property_function(
    activity_checkpoint="amp_design/best_new_4.pth",
    toxicity_checkpoint="personalization/checkpoints/toxicity_head.pth",
    stability_checkpoint="personalization/checkpoints/stability_head.pth",
    esm_model_size="650M",
    device="cuda",
    max_length=100,
)

# Use it
sequences = ["GIGKFLHSAKKFGKAFVGEIMNS", "KKLLPIVKKK"]
properties = property_fn(sequences)  # (2, 4) tensor
# properties[:, 0] = activity
# properties[:, 1] = toxicity
# properties[:, 2] = stability
# properties[:, 3] = normalized length
```

### 3. Personas (User Preferences)

Define user preferences as weight vectors:

```python
from personalization import get_persona, Persona

# Use pre-defined persona
persona = get_persona("SafetyFirst")
print(persona.explain())
# Output:
# Persona: SafetyFirst
# Description: Strongly emphasizes safety (low toxicity)...
# Property Weights:
#   activity:     +0.50 (prefers high)
#   toxicity:     -1.00 (strongly avoids high)
#   stability:    +0.50 (prefers high)
#   length:       -0.20 (slightly prefers shorter)

# Or create custom persona
from personalization import create_custom_persona

my_persona = create_custom_persona(
    name="MyCustomUser",
    activity_weight=0.8,
    toxicity_weight=-0.6,
    stability_weight=0.7,
    length_weight=-0.3,
    description="Custom therapeutic peptide preferences"
)
```

**Pre-defined personas:**
- `PotencyMaximizer`: Maximum activity, neutral on toxicity
- `SafetyFirst`: Strong toxicity penalty, moderate activity
- `BalancedDesigner`: Balanced trade-offs (recommended default)
- `StabilityFocused`: Maximize stability for shelf life
- `ShortPeptideFan`: Strong preference for short sequences
- `TherapeuticOptimizer`: Optimized for therapeutic use
- `ResearchExplorer`: Activity discovery, explore diverse space

### 4. Personalized Reward Computation

The reward is a simple dot product:

```python
from personalization import compute_personalized_reward

# Get properties
properties = property_fn(sequences)  # (batch_size, 4)

# Get persona
persona = get_persona("BalancedDesigner")

# Compute rewards
rewards = compute_personalized_reward(properties, persona)  # (batch_size,)
```

**Example calculation:**
```
Sequence: "GIGKFLHSAKKFGKAFVGEIMNS"
Properties: [p_act=0.90, p_tox=0.30, p_stab=0.70, p_len=0.23]
Persona (BalancedDesigner): w = [0.7, -0.5, 0.6, -0.1]

Reward = w^T · g
       = 0.7 × 0.90 + (-0.5) × 0.30 + 0.6 × 0.70 + (-0.1) × 0.23
       = 0.63 - 0.15 + 0.42 - 0.023
       = 0.877
```

## Integration with GRPO/PPO Training

### Option 1: Pure Personalized Rewards

Replace the base reward function with personalized rewards:

```python
from personalization import create_unified_property_function, get_persona
from amp_design.personalized_grpo_v2 import create_personalized_reward_fn

# Setup
property_fn = create_unified_property_function(...)
persona = get_persona("SafetyFirst")

# Create reward function
reward_fn = create_personalized_reward_fn(property_fn, persona, device="cuda")

# Use in GRPO/PPO
rewards, mask = reward_fn(sequences)
```

### Option 2: Blended Rewards

Gradually introduce personalization by blending with base rewards:

```python
from amp_design.personalized_grpo_v2 import create_blended_reward_fn

# Base reward function (e.g., classifier)
def base_reward_fn(sequences):
    return reward_amp_cls(sequences, ...)

# Blended reward: 50% base + 50% personalized
blended_reward_fn = create_blended_reward_fn(
    property_function=property_fn,
    persona=persona,
    base_reward_fn=base_reward_fn,
    blend_weight=0.5,  # 0 = all base, 1 = all personalized
    device="cuda"
)

# Use in training
rewards, mask = blended_reward_fn(sequences)
```

## Interpretability

### Explain Individual Rewards

```python
from personalization import explain_reward

# For a single sequence
sequence = "GIGKFLHSAKKFGKAFVGEIMNS"
properties = property_fn([sequence])[0]  # (4,)
persona = get_persona("SafetyFirst")

explanation = explain_reward(properties, persona, sequence)
print(explanation)
```

**Output:**
```
Sequence: GIGKFLHSAKKFGKAFVGEIMNS
Persona: SafetyFirst
Total Reward: 0.4250

Property Contributions:
  Activity    :  0.900 ×   0.50 =  0.4500
  Toxicity    :  0.300 ×  -1.00 = -0.3000
  Stability   :  0.700 ×   0.50 =  0.3500
  Length      :  0.230 ×  -0.20 = -0.0460

  Total       :               =  0.4540
```

### Compare Personas

```python
from amp_design.personalized_grpo_v2 import compare_personas_on_sequences

sequences = ["GIGKFLHSAKKFGKAFVGEIMNS", "KKLLPIVKKK", "AAAAGGGGAAAA"]
df = compare_personas_on_sequences(
    sequences, 
    property_fn,
    persona_names=["PotencyMaximizer", "SafetyFirst", "BalancedDesigner"]
)
print(df)
```

**Output:**
```
                  sequence  PotencyMaximizer_reward  SafetyFirst_reward  BalancedDesigner_reward
GIGKFLHSAKKFGKAFVGEIMNS                    0.969              0.454                     0.877
            KKLLPIVKKK                    0.850              0.620                     0.750
          AAAAGGGGAAAA                    0.210              0.180                     0.195
```

## Complete Example Workflow

```python
# 1. Setup property function (once per session)
from personalization import create_unified_property_function

property_fn = create_unified_property_function(
    activity_checkpoint="amp_design/best_new_4.pth",
    toxicity_checkpoint="personalization/checkpoints/toxicity_head.pth",
    stability_checkpoint="personalization/checkpoints/stability_head.pth",
    esm_model_size="650M",
    device="cuda",
)

# 2. Choose persona
from personalization import get_persona

persona = get_persona("SafetyFirst")
print(persona.explain())

# 3. Create reward function
from amp_design.personalized_grpo_v2 import create_personalized_reward_fn

reward_fn = create_personalized_reward_fn(property_fn, persona, "cuda")

# 4. Use in RL training
def train_policy():
    # ... policy initialization ...
    
    for batch in dataloader:
        sequences = generate_sequences(policy, prompts)
        rewards, mask = reward_fn(sequences)
        
        # GRPO/PPO update
        loss = compute_policy_loss(sequences, rewards, mask)
        loss.backward()
        optimizer.step()

# 5. Switch personas mid-training (optional)
persona = get_persona("BalancedDesigner")
reward_fn = create_personalized_reward_fn(property_fn, persona, "cuda")
# Continue training with different preferences!
```

## Adding New Users

### Option 1: Define Weight Vector Directly

```python
from personalization import create_custom_persona, register_persona

# Create custom persona
custom_persona = create_custom_persona(
    name="IndustrialProduction",
    activity_weight=0.6,
    toxicity_weight=-0.7,
    stability_weight=0.9,  # Critical for shelf life
    length_weight=-0.5,    # Shorter = cheaper synthesis
    description="Optimized for large-scale production"
)

# Register it
register_persona(custom_persona)

# Now you can use it like any pre-defined persona
persona = get_persona("IndustrialProduction")
```

### Option 2: Learn from Preferences (Future)

If you have real pairwise preference data from a user:

```python
# Collect preferences
preferences = [
    (seq_i, seq_j, user_prefers_i),  # tuples of (seq_a, seq_b, label)
    ...
]

# Compute properties for all sequences
all_seqs = list(set([p[0] for p in preferences] + [p[1] for p in preferences]))
properties_dict = {seq: property_fn([seq])[0] for seq in all_seqs}

# Learn weights using Bradley-Terry model
# (optimization problem: find w that best explains preferences)
from personalization.learn_weights import learn_user_weights_from_preferences

w_learned = learn_user_weights_from_preferences(
    properties_dict,
    preferences,
    num_properties=4,
)

# Create persona from learned weights
learned_persona = Persona(
    name="UserX",
    weights=w_learned,
    description="Learned from user preferences"
)
```

## FAQ

### Q: Do I need to retrain property heads for each user?

**No!** That's the whole point. Train each property head once:
- Activity: Use existing `best_new_4.pth`
- Toxicity: Train once on ToxDL2 (14K sequences)
- Stability: Train once on stability dataset or use EsmTherm

Then define users with 4 weights each.

### Q: How do I handle missing property predictions?

If you don't have a trained model for a property (e.g., toxicity), you can:

1. **Use placeholder:** Train placeholder model with `train_stability.py --mode placeholder`
2. **Set weight to zero:** If a persona has weight 0 for that property, it won't affect rewards
3. **Use proxy:** For stability, could use simple heuristics initially

### Q: Can I use different ESM models?

Yes! The architecture supports ESM2-650M (33 layers) or ESM2-8M (6 layers):

```python
property_fn = create_unified_property_function(
    ...,
    esm_model_size="8M",  # Faster, lighter
)
```

ESM2-8M is faster but less accurate. Use 650M for best results.

### Q: How do I integrate with existing GRPO code?

Replace the reward function in your GRPO training loop:

```python
# Old
rewards, mask = reward_amp_cls(sequences, esm_model, ...)

# New
rewards, mask = personalized_reward_fn(sequences)
```

The interface is the same: `(sequences) -> (rewards, mask)`.

### Q: Can I train a model that's conditioned on user embeddings?

Yes! Instead of explicit weights, you can train:

```python
R(x, u) = f_θ(g(x), e_u)
```

where `e_u` is a learnable user embedding. This is more flexible but less interpretable. See `personalization/user_conditioned_model.py` (future work).

## Performance Tips

1. **Batch property computation:** Compute properties for multiple sequences at once
2. **Cache ESM embeddings:** If evaluating same sequences repeatedly
3. **Use ESM2-8M:** For faster iteration during development
4. **GPU utilization:** Property heads are small, ensure ESM model is on GPU

## Next Steps

1. **Train toxicity head:** `python personalization/train_toxicity.py`
2. **Create stability head:** `python personalization/train_stability.py --mode placeholder`
3. **Test property function:** `python amp_design/personalized_grpo_v2.py --demo`
4. **Integrate with GRPO:** Modify your training loop to use personalized rewards
5. **Experiment with personas:** Try different weight combinations

## References

- ToxDL2: Protein toxicity prediction ([paper](https://www.nature.com/articles/...))
- EsmTherm: Stability prediction ([paper](https://www.nature.com/articles/s41586-023-06328-6))
- ESM-2: Protein language models ([GitHub](https://github.com/facebookresearch/esm))

