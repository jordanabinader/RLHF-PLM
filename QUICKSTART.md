# Quick Start Guide: Shared Property Model for Personalized RLHF

## Installation

```bash
# Clone repository
cd /path/to/RLHF-PLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install fair-esm (protein language model)
pip install fair-esm
```

## Step 1: Train Property Heads (One-time Setup)

### Train Toxicity Head

```bash
python personalization/train_toxicity.py \
    --data-dir ToxDL2-main/data \
    --output-dir personalization/checkpoints \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

**Expected output:** `personalization/checkpoints/toxicity_head.pth`

**Training time:** ~2-3 hours on GPU (14K sequences)

### Create Stability Head

For quick start, use placeholder mode:

```bash
python personalization/train_stability.py \
    --mode placeholder \
    --output-dir personalization/checkpoints
```

**Expected output:** `personalization/checkpoints/stability_head.pth`

**Training time:** Instant (placeholder model)

> **Note:** For production use, train on real stability data or load EsmTherm checkpoint

### Activity Head

Already available: `amp_design/best_new_4.pth` (pre-trained)

## Step 2: Run Demo

Test the complete pipeline:

```bash
python examples/property_model_demo.py
```

**What it shows:**
- Property computation for example sequences
- Rewards from different personas
- Interpretability (explaining individual rewards)
- Comparison across personas

## Step 3: Test Integration with GRPO

```bash
python amp_design/personalized_grpo_v2.py \
    --persona SafetyFirst \
    --demo \
    --device cuda
```

**What it does:**
- Loads unified property function
- Creates personalized reward function
- Analyzes example sequences
- Shows how different personas rank sequences

## Usage in Your Code

### Basic Usage

```python
from personalization import (
    create_unified_property_function,
    get_persona,
    compute_personalized_reward,
)

# 1. Setup (once per session)
property_fn = create_unified_property_function(
    activity_checkpoint="amp_design/best_new_4.pth",
    toxicity_checkpoint="personalization/checkpoints/toxicity_head.pth",
    stability_checkpoint="personalization/checkpoints/stability_head.pth",
    esm_model_size="650M",
    device="cuda",
)

# 2. Choose persona
persona = get_persona("SafetyFirst")

# 3. Compute properties and rewards
sequences = ["GIGKFLHSAKKFGKAFVGEIMNS", "KKLLPIVKKK"]
properties = property_fn(sequences)  # (2, 4) tensor
rewards = compute_personalized_reward(properties, persona)  # (2,) tensor

print(f"Rewards: {rewards}")
```

### Integration with RL Training

```python
from amp_design.personalized_grpo_v2 import create_personalized_reward_fn

# Create reward function
reward_fn = create_personalized_reward_fn(property_fn, persona, device="cuda")

# Use in GRPO/PPO training
for batch in dataloader:
    sequences = generate_sequences(policy, prompts)
    rewards, mask = reward_fn(sequences)
    
    # Compute loss and update policy
    loss = compute_policy_loss(sequences, rewards, mask)
    loss.backward()
    optimizer.step()
```

### Switching Personas

```python
# Train with different personas
personas = ["PotencyMaximizer", "SafetyFirst", "BalancedDesigner"]

for persona_name in personas:
    persona = get_persona(persona_name)
    reward_fn = create_personalized_reward_fn(property_fn, persona, "cuda")
    
    # Train policy with this persona
    train_policy(policy, reward_fn, num_epochs=10)
    
    # Save persona-specific checkpoint
    torch.save(policy.state_dict(), f"checkpoints/policy_{persona_name}.pt")
```

## Available Personas

| Persona | Activity | Toxicity | Stability | Length | Description |
|---------|----------|----------|-----------|--------|-------------|
| **PotencyMaximizer** | +1.0 | 0.0 | +0.3 | 0.0 | Maximum activity |
| **SafetyFirst** | +0.5 | -1.0 | +0.5 | -0.2 | Strong toxicity penalty |
| **BalancedDesigner** | +0.7 | -0.5 | +0.6 | -0.1 | Balanced (recommended) |
| **StabilityFocused** | +0.4 | -0.3 | +1.0 | 0.0 | Maximize stability |
| **ShortPeptideFan** | +0.6 | -0.4 | +0.3 | -0.8 | Prefer short sequences |
| **TherapeuticOptimizer** | +0.6 | -0.9 | +0.8 | -0.3 | For therapeutic use |
| **ResearchExplorer** | +0.8 | 0.0 | +0.2 | 0.0 | Activity discovery |

## Creating Custom Personas

```python
from personalization import create_custom_persona

my_persona = create_custom_persona(
    name="IndustrialProduction",
    activity_weight=0.6,
    toxicity_weight=-0.7,
    stability_weight=0.9,   # Critical for shelf life
    length_weight=-0.5,     # Shorter = cheaper synthesis
    description="Optimized for large-scale production"
)

# Use it
rewards = compute_personalized_reward(properties, my_persona)
```

## Interpreting Results

```python
from personalization import explain_reward

# Explain why a sequence got its reward
sequence = "GIGKFLHSAKKFGKAFVGEIMNS"
properties = property_fn([sequence])[0]
persona = get_persona("SafetyFirst")

print(explain_reward(properties, persona, sequence))
```

**Output:**
```
Sequence: GIGKFLHSAKKFGKAFVGEIMNS
Persona: SafetyFirst
Total Reward: 0.4540

Property Contributions:
  Activity    :  0.900 ×   0.50 =  0.4500
  Toxicity    :  0.300 ×  -1.00 = -0.3000  ← Strong penalty!
  Stability   :  0.700 ×   0.50 =  0.3500
  Length      :  0.230 ×  -0.20 = -0.0460

  Total       :               =  0.4540
```

## Common Issues

### Issue: "Checkpoint not found"

**Solution:** Train the property heads first (see Step 1)

### Issue: "CUDA out of memory"

**Solution:** Use smaller ESM model or reduce batch size

```python
property_fn = create_unified_property_function(
    ...,
    esm_model_size="8M",  # Faster, less memory
)
```

### Issue: "ImportError: No module named 'esm'"

**Solution:** Install fair-esm

```bash
pip install fair-esm
```

## File Structure

```
RLHF-PLM/
├── personalization/
│   ├── property_models.py          # Property head architectures
│   ├── unified_property_fn.py      # Unified g(x) function
│   ├── personas.py                 # User weight vectors
│   ├── train_toxicity.py           # Training script
│   ├── train_stability.py          # Training script
│   └── checkpoints/
│       ├── toxicity_head.pth       # Trained model
│       └── stability_head.pth      # Trained model
├── amp_design/
│   ├── personalized_grpo_v2.py     # GRPO integration
│   └── best_new_4.pth              # Activity classifier
├── examples/
│   └── property_model_demo.py      # Complete demo
├── SHARED_PROPERTY_SUMMARY.md      # Implementation summary
├── personalization/PROPERTY_MODEL_GUIDE.md  # Detailed guide
└── requirements.txt                # Dependencies
```

## Documentation

- **Architecture Guide:** `personalization/PROPERTY_MODEL_GUIDE.md` (620 lines)
- **Implementation Summary:** `SHARED_PROPERTY_SUMMARY.md`
- **API Reference:** Docstrings in all modules
- **Demo Script:** `examples/property_model_demo.py`

## Performance

- **Property computation:** ~50ms per batch (10 sequences) on GPU
- **Reward computation:** <1ms (simple dot product)
- **Adding new user:** Instant (just define 4 weights)
- **Training property head:** ~2-3 hours once

## Next Steps

1. ✅ Train property heads
2. ✅ Run demo to verify setup
3. ✅ Integrate with your RL training loop
4. ✅ Experiment with different personas
5. ✅ Create custom personas for your needs

## Support

- **Issues:** Open an issue on GitHub
- **Documentation:** See `PROPERTY_MODEL_GUIDE.md`
- **Examples:** Run `examples/property_model_demo.py`

## Citation

If you use this code, please cite:

```bibtex
@software{rlhf_plm_shared_property,
  title={Shared Property Model for Personalized RLHF in Protein Design},
  year={2024},
  url={https://github.com/your-repo/RLHF-PLM}
}
```

---

**Questions?** Check `personalization/PROPERTY_MODEL_GUIDE.md` for detailed documentation.

