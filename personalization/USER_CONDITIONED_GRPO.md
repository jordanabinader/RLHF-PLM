

# User-Conditioned GRPO for Personalized Protein Design

## Overview

This implementation provides a **single policy** that generates protein sequences conditioned on user preferences, replacing the need for training multiple per-persona policies.

### Key Benefits

1. **Single Model Training**: Train once, use for all personas
2. **Efficient**: No need to retrain for new user preferences
3. **Interpretable**: Explicit property weights explain preferences
4. **Enforced Validity**: Hard constraints ensure biologically reasonable sequences
5. **Lightweight Personalization**: Only 4-dimensional weight vectors per user

## Architecture

```
Sequence Generation:
  User Weights w^(u) → Projector → User Embeddings
                                         ↓
  Prompt → Policy (conditioned on user embeddings) → Generated Sequence x

Reward Computation:
  Sequence x → Property Function g(x) → [p_act, p_tox, p_stab, p_len]
                                                ↓
                          Validity Check → Valid? → R^(u)(x) = w^(u)^T · g(x)
                                              ↓
                                          Invalid → R_penalty = -10
```

### Components

#### 1. User Context Projector
- **Input**: 4D weight vector `w^(u) = [w_act, w_tox, w_stab, w_len]`
- **Output**: High-dimensional embedding (256D)
- **Purpose**: Projects user preferences into policy's embedding space

#### 2. Hybrid Reward Function
- **Validity Constraints** (hard filters):
  - Only canonical amino acids (20 standard AAs)
  - Length between 8-50 residues
  - Minimum net charge (default: 0 for AMPs)
  - No excessive repeats (max 50% of any single AA)
- **Personalized Rewards** (for valid sequences):
  - `R^(u)(x) = w^(u)^T · g(x)`
  - Where `g(x) = [p_act, p_tox, p_stab, p_len]`

#### 3. Property Function
- **Activity** (`p_act`): AMP classification score
- **Toxicity** (`p_tox`): Toxicity prediction
- **Stability** (`p_stab`): Thermostability prediction
- **Length** (`p_len`): Normalized sequence length

## Usage

### 1. Training with Single Persona

Train a policy optimized for one specific persona:

```bash
python amp_design/grpo.py \
  --base-model-path progen2hf/progen2-small \
  --tokenizer-path progen2hf/ \
  --classifier-checkpoint amp_design/best_new_4.pth \
  --toxicity-checkpoint personalization/checkpoints/toxicity_head.pth \
  --stability-checkpoint personalization/checkpoints/stability_head.pth \
  --use-personalization \
  --persona-name BalancedDesigner \
  --persona-cycle-mode single \
  --output-dir grpo_runs/balanced_designer \
  --epochs 10 \
  --batch-size 32
```

### 2. Training with Multi-Persona Cycling

Train a single policy that adapts to multiple personas:

```bash
python amp_design/grpo.py \
  --base-model-path progen2hf/progen2-small \
  --tokenizer-path progen2hf/ \
  --classifier-checkpoint amp_design/best_new_4.pth \
  --toxicity-checkpoint personalization/checkpoints/toxicity_head.pth \
  --stability-checkpoint personalization/checkpoints/stability_head.pth \
  --use-personalization \
  --persona-cycle-mode random \
  --output-dir grpo_runs/multi_persona \
  --epochs 20 \
  --batch-size 32
```

**Cycling Modes**:
- `single`: Use one persona throughout training
- `random`: Randomly sample a persona for each batch
- `round_robin`: Cycle through personas in order

### 3. Evaluating Trained Policy

Evaluate across all personas:

```bash
python personalization/evaluate_user_conditioned_policy.py \
  --checkpoint grpo_runs/multi_persona/final_model \
  --tokenizer-path progen2hf/ \
  --activity-checkpoint amp_design/best_new_4.pth \
  --toxicity-checkpoint personalization/checkpoints/toxicity_head.pth \
  --stability-checkpoint personalization/checkpoints/stability_head.pth \
  --num-sequences 200 \
  --output-dir evaluation_results/
```

## Available Personas

Each persona has unique weights `w^(u) = [w_act, w_tox, w_stab, w_len]`:

| Persona | Activity | Toxicity | Stability | Length | Description |
|---------|----------|----------|-----------|--------|-------------|
| **PotencyMaximizer** | +1.0 | 0.0 | +0.3 | 0.0 | Prioritizes activity above all |
| **SafetyFirst** | +0.5 | -1.0 | +0.5 | -0.2 | Strong penalty for toxicity |
| **BalancedDesigner** | +0.7 | -0.5 | +0.6 | -0.1 | Balanced trade-offs |
| **StabilityFocused** | +0.4 | -0.3 | +1.0 | 0.0 | Maximizes stability |
| **ShortPeptideFan** | +0.6 | -0.4 | +0.3 | -0.8 | Prefers short sequences |

## Creating Custom Personas

```python
from personalization import create_custom_persona

# Define your persona
my_persona = create_custom_persona(
    name="MyCustomPersona",
    w_activity=0.8,      # High activity
    w_toxicity=-0.6,     # Moderate toxicity penalty
    w_stability=0.4,     # Some stability preference
    w_length=-0.3,       # Slight preference for shorter sequences
    description="My custom preferences"
)

# Use in training or evaluation
```

## Testing

### Unit Tests

Test individual components:

```bash
python tests/test_user_conditioned_grpo.py
```

Tests:
- Validity constraints
- Net charge calculation
- Batch validation
- Hybrid reward function
- User context projection
- Persona weight differences

### Integration Test

Test end-to-end pipeline:

```bash
python tests/test_integration_user_conditioned.py
```

Verifies:
- Property function computation
- Hybrid reward with validity constraints
- Persona-based reward calculation
- User context projection

## Implementation Details

### Validity Constraints

**File**: `personalization/validity.py`

```python
def is_sequence_valid(sequence: str, min_charge: float = 0.0) -> bool:
    # 1. Only canonical amino acids
    if not all(aa in CANONICAL_AAS for aa in sequence):
        return False
    
    # 2. Length constraints (8-50 residues)
    if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):
        return False
    
    # 3. Minimum charge (for AMPs)
    if calculate_net_charge(sequence) < min_charge:
        return False
    
    # 4. No excessive repeats (max 50% of any AA)
    max_repeat = max(Counter(sequence).values())
    if max_repeat / len(sequence) > 0.5:
        return False
    
    return True
```

### Hybrid Reward

**File**: `personalization/hybrid_reward.py`

```python
def create_hybrid_reward_fn(property_function, persona, penalty=-10.0):
    def reward_fn(sequences):
        # Check validity
        valid_mask = validate_sequences(sequences)
        
        # Compute personalized rewards
        properties = property_function(sequences)
        persona_rewards = compute_personalized_reward(properties, persona)
        
        # Apply conditional reward
        rewards = torch.where(
            valid_mask,
            persona_rewards,  # Valid: use personalized reward
            torch.full_like(persona_rewards, penalty)  # Invalid: penalty
        )
        
        return rewards, valid_mask
    
    return reward_fn
```

### User-Conditioned Policy

**File**: `personalization/user_conditioned_policy.py`

The policy wrapper:
1. Projects user weights to high-dimensional space
2. Stores user context for generation
3. Conditions generation on user embeddings

```python
class UserConditionedPolicyWrapper(nn.Module):
    def __init__(self, base_policy, user_dim=4, projection_dim=256):
        super().__init__()
        self.base_policy = base_policy
        self.user_projector = UserContextProjector(
            user_dim=user_dim,
            output_dim=projection_dim
        )
    
    def generate(self, input_ids, user_context, **kwargs):
        # Project user context
        user_embed = self.user_projector(user_context)
        
        # Store for generation
        self.current_user_context = user_embed
        
        # Generate with base policy
        return self.base_policy.generate(input_ids, **kwargs)
```

## Training Tips

1. **Start with Single Persona**: Verify the system works before multi-persona training
2. **Adjust Penalty**: If too few valid sequences, reduce `--reward-penalty`
3. **Monitor Validity Rate**: Should be >70% for effective training
4. **Batch Size**: Larger batches (64-128) improve stability
5. **Learning Rate**: 2e-5 is a good starting point
6. **Checkpoint Frequency**: Save every 25-50 steps for analysis

## Troubleshooting

### Issue: Low validity rate (<50%)

**Solution**: Adjust constraints
```bash
--min-charge -1.0  # More permissive charge requirement
--reward-penalty -5.0  # Smaller penalty
```

### Issue: Policy ignores persona

**Symptoms**: All personas generate similar sequences

**Solutions**:
1. Increase persona weight magnitudes
2. Use longer training (more epochs)
3. Ensure multi-persona cycling is enabled
4. Check that user context is passed to generation

### Issue: OOM errors

**Solutions**:
```bash
--batch-size 16  # Reduce batch size
--num-candidates 4  # Fewer candidates per prompt
--max-new-tokens 30  # Shorter sequences
```

## Performance Metrics

Expected results after training:

| Metric | Target |
|--------|--------|
| Validity Rate | >70% |
| Sequence Uniqueness | >80% |
| Persona Differentiation | Different property distributions per persona |
| Training Time | ~2-4 hours (10 epochs, 1 GPU) |

## Future Enhancements

1. **Advanced User Context Integration**: Cross-attention with user embeddings
2. **Dynamic Weight Adjustment**: Learn optimal weights from feedback
3. **Multi-Objective Optimization**: Pareto-optimal generation
4. **Active Learning**: Iterative refinement with user feedback
5. **Transfer Learning**: Apply to other protein design tasks

## References

- Shared Property Model: `personalization/PROPERTY_MODEL_GUIDE.md`
- Personas: `personalization/personas.py`
- GRPO Training: `amp_design/grpo.py`
- Property Models: `personalization/property_models.py`

## Support

For issues or questions:
1. Check integration test: `python tests/test_integration_user_conditioned.py`
2. Review logs for error messages
3. Verify all checkpoints exist and are compatible
4. Check GPU memory usage

---

**Author**: RLHF-PLM Team  
**Last Updated**: December 2024  
**Version**: 1.0

