# Personalized RLHF Integration Guide

This guide explains where and how personalization modules have been integrated into the RLHF-PLM codebase.

## 🗺️ Architecture Overview

```
RLHF-PLM/
│
├── personalization/                    # Core personalization module
│   ├── __init__.py
│   ├── synthetic_users.py              # User definitions & property computation
│   ├── preference_generation.py        # Pairwise preference generation
│   ├── preference_reward_model.py      # Bradley-Terry reward models
│   ├── personalized_trainer.py         # Base classes for personalized RL
│   └── README.md                       # Detailed module documentation
│
├── amp_design/                         # Antimicrobial peptide design
│   ├── personalized_grpo.py            # ✨ NEW: Personalized AMP training
│   ├── grpo.py                         # Original GRPO (can be extended)
│   ├── ppo.py                          # Original PPO (can be extended)
│   └── reward.py                       # Base reward functions
│
├── antibody_mutation/                  # Antibody mutation design
│   ├── personalized_mutation.py        # ✨ NEW: Personalized antibody training
│   └── mutation_policy_grpo.py         # Original mutation policy
│
├── kinase_mutation/                    # Kinase mutation design
│   ├── personalized_kinase.py          # ✨ NEW: Personalized kinase training
│   └── PhoQ_env.py                     # Original kinase environment
│
└── examples/                           # ✨ NEW: Example scripts
    └── personalized_amp_example.py     # Complete end-to-end example
```

## 🔧 Integration Points by Task

### 1. AMP Design (`amp_design/`)

**File**: `amp_design/personalized_grpo.py`

**Key Functions**:
- `build_amp_feature_matrix()`: Extracts ESM embeddings + properties
- `train_personalized_amp_reward_model()`: Trains user-specific reward model
- `create_personalized_amp_reward_fn()`: Creates blended reward function

**Integration with existing code**:
```python
# In your GRPO training loop (amp_design/grpo.py):
from personalized_grpo import create_personalized_amp_reward_fn
from personalization.synthetic_users import define_synthetic_users_amp

# Select a user
users = define_synthetic_users_amp()
user = users[0]  # e.g., PotencyMaximizer

# Train personalized reward model (do this once before RL training)
reward_model = train_personalized_amp_reward_model(
    sequences=training_sequences,
    user=user,
    esm_model=esm_model,
    batch_converter=batch_converter,
    alphabet=alphabet,
    classifier=classifier,
)

# Create personalized reward function
personalized_reward_fn = create_personalized_amp_reward_fn(
    user=user,
    reward_model=reward_model,
    esm_model=esm_model,
    batch_converter=batch_converter,
    alphabet=alphabet,
    classifier=classifier,
    blend_weight=0.5,  # 50% classifier, 50% personalized
)

# Use in your GRPO loop
def reward_fn(seqs):
    return personalized_reward_fn(clean_sequences(seqs))

# Continue with GRPO training as usual...
groups = trainer.create_grpo_groups(prompts, candidates_list, reward_fn)
```

**Where to modify**:
1. `grpo.py` line ~604: Replace `reward_fn` with personalized version
2. `ppo.py` line ~65: Replace `reward_fn` in `prepare_reward_model()`

### 2. Antibody Mutation (`antibody_mutation/`)

**File**: `antibody_mutation/personalized_mutation.py`

**Key Functions**:
- `build_antibody_feature_matrix()`: Extracts ESM embeddings
- `train_personalized_antibody_reward_model()`: Trains reward model
- `create_personalized_antibody_reward_fn()`: Creates reward function

**Integration with existing code**:
```python
# In mutation_policy_grpo.py:
from personalized_mutation import (
    train_personalized_antibody_reward_model,
    create_personalized_antibody_reward_fn,
)

# In MutationGRPOTrainer.__init__():
# After loading base models, train personalized reward
self.personalized_reward_model = train_personalized_antibody_reward_model(
    sequences=training_sequences,
    user=selected_user,
    model=self.policy.base_model,
    tokenizer=tokenizer,
    base_reward_model=self.reward,
)

# In _train_batch():
# Replace or blend with self.reward
personalized_reward_fn = create_personalized_antibody_reward_fn(
    user=self.user,
    preference_model=self.personalized_reward_model,
    base_model=self.policy.base_model,
    tokenizer=tokenizer,
)

# Use for computing rewards
rew_mut, _ = personalized_reward_fn(mutated_sequences)
```

**Where to modify**:
1. `mutation_policy_grpo.py` line ~204: Add personalized reward model to trainer
2. `mutation_policy_grpo.py` line ~327-337: Use personalized rewards

### 3. Kinase Mutation (`kinase_mutation/`)

**File**: `kinase_mutation/personalized_kinase.py`

**Key Functions**:
- `build_kinase_feature_matrix()`: Extracts features
- `train_personalized_kinase_reward_model()`: Trains reward model
- `create_personalized_kinase_reward_fn()`: Creates reward function

**Integration with existing code**:
```python
# In PhoQ_env.py:
from personalized_kinase import (
    train_personalized_kinase_reward_model,
    create_personalized_kinase_reward_fn,
)

# Before creating environment, train personalized reward model
from personalization.synthetic_users import define_synthetic_users_kinase

users = define_synthetic_users_kinase()
user = users[0]

# Load kinase data
sequences, fitness_scores = load_kinase_data()

personalized_reward_model = train_personalized_kinase_reward_model(
    sequences=sequences,
    fitness_scores=fitness_scores,
    user=user,
    tokenizer=tokenizer,
    model=esm_model,
)

# Create personalized reward function
personalized_reward_fn = create_personalized_kinase_reward_fn(
    preference_model=personalized_reward_model,
    tokenizer=tokenizer,
    model=esm_model,
)

# In PhoQEnv._get_reward():
# Replace fitness lookup with personalized reward
score_personalized = personalized_reward_fn([protein_string])[0].item()
```

**Where to modify**:
1. `PhoQ_env.py` line ~98-122: Modify `_get_reward()` method
2. Create personalized reward model before environment initialization

## 🎯 Workflow for Adding Personalization

### Step 1: Pre-train Preference Reward Models

Before running RL training, pre-train personalized reward models:

```bash
# AMP Design
python amp_design/personalized_grpo.py \
    --classifier-checkpoint path/to/classifier.pt \
    --output-dir personalized_rewards/amp

# Antibody Mutation
python antibody_mutation/personalized_mutation.py \
    --data-file data/antibodies.csv \
    --output-dir personalized_rewards/antibody

# Kinase Mutation
python kinase_mutation/personalized_kinase.py \
    --data-file data/PhoQ.csv \
    --output-dir personalized_rewards/kinase
```

### Step 2: Modify RL Training Scripts

Load and use the pre-trained reward models in your RL training:

```python
# Load personalized reward model
reward_model = PreferenceRewardModel(input_dim=...)
reward_model.load_state_dict(torch.load("personalized_rewards/user_model.pt"))

# Create wrapper or direct reward function
personalized_reward_fn = create_reward_function(reward_model, ...)

# Use in RL training
# ... continue with PPO/GRPO/DPO training
```

### Step 3: Run RL Training with Different Users

Train separate policies for different users:

```bash
# Train for "SafetyFirst" user
python amp_design/grpo.py \
    --user-preference safetyfirst \
    --reward-model personalized_rewards/amp/safetyfirst.pt \
    --output-dir results/amp_safetyfirst

# Train for "PotencyMaximizer" user
python amp_design/grpo.py \
    --user-preference potencymaximizer \
    --reward-model personalized_rewards/amp/potencymaximizer.pt \
    --output-dir results/amp_potencymaximizer
```

## 📝 Customization Guide

### Adding New Synthetic Users

Edit `personalization/synthetic_users.py`:

```python
def define_synthetic_users_amp():
    users = [
        # ... existing users ...
        
        # Add your custom user
        SyntheticUser(
            name="MyCustomUser",
            weights={
                "activity_score": +0.8,
                "toxicity_score": -0.6,
                "stability_score": +0.4,
                "custom_property": +1.0,  # Your custom property
            },
            description="My custom preferences",
        ),
    ]
    return users
```

### Adding New Properties

Define property computation functions:

```python
# In synthetic_users.py or your task-specific file
def compute_my_property(seq: str) -> float:
    """Compute my custom property."""
    # Your computation logic
    return value

# Use in property computation
property_functions = {
    "my_property": compute_my_property,
    # ... other properties
}

df = compute_property_scores(
    sequences=sequences,
    property_functions=property_functions,
)
```

### Tuning Blend Weight

The `blend_weight` parameter controls the balance between base rewards and personalized rewards:

- `blend_weight=0.0`: Only base reward (e.g., classifier)
- `blend_weight=0.5`: Equal blend
- `blend_weight=1.0`: Only personalized reward

Recommended starting points:
- **Early training**: 0.2-0.3 (rely more on base reward)
- **Mid training**: 0.5 (balanced)
- **Late training**: 0.7-0.8 (rely more on personalized preferences)

## 🔬 Example: Complete AMP Personalization

```bash
# 1. Train personalized reward models for all users
python amp_design/personalized_grpo.py \
    --classifier-checkpoint amp_design/best_new_4.pth \
    --output-dir personalized_rewards/amp \
    --esm-mode 8M \
    --num-pairs 5000

# 2. Run GRPO with personalized rewards (modify grpo.py as shown above)
# Create a wrapper script:
cat > amp_design/run_personalized_grpo.sh << 'EOF'
#!/bin/bash

USERS=("potencymaximizer" "safetyfirst" "balanceddesigner")

for user in "${USERS[@]}"; do
    echo "Training GRPO for user: $user"
    python amp_design/grpo.py \
        --base-model-path progen2hf/models \
        --tokenizer-path progen2hf \
        --classifier-checkpoint amp_design/best_new_4.pth \
        --personalized-reward personalized_rewards/amp/reward_model_${user}.pt \
        --output-dir results/amp_grpo_${user} \
        --exp-name grpo_${user}
done
EOF

chmod +x amp_design/run_personalized_grpo.sh
./amp_design/run_personalized_grpo.sh
```

## 📊 Evaluation

Compare sequences generated for different users:

```python
import pandas as pd

results = {}
for user in ["potencymaximizer", "safetyfirst", "balanceddesigner"]:
    # Load generated sequences
    seqs = load_sequences(f"results/amp_grpo_{user}/sequences.txt")
    
    # Evaluate properties
    properties = evaluate_properties(seqs)
    results[user] = properties

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
print(comparison_df)
```

## 🚀 Next Steps

1. **Experiment with user definitions**: Adjust weights to match your specific needs
2. **Collect real human preferences**: Replace synthetic preferences with actual user feedback
3. **Multi-objective optimization**: Use multiple users to explore the Pareto front
4. **Active learning**: Iteratively refine preferences based on generated sequences

## 📚 Additional Resources

- See `personalization/README.md` for detailed API documentation
- See `examples/personalized_amp_example.py` for a complete working example
- See original task READMEs for baseline RL training instructions

## ❓ FAQ

**Q: Can I use the same reward model for multiple RL runs?**  
A: Yes! Train once, use many times. Reward models are deterministic.

**Q: How do I handle multiple users simultaneously?**  
A: Train a single model with `use_user_conditioning=True` and pass `user_id` at inference.

**Q: What if my properties are expensive to compute?**  
A: Compute once, cache in a CSV, and load during preference generation.

**Q: Can I use real human feedback instead of synthetic?**  
A: Absolutely! Just format it as (idx_a, idx_b, pref) and use `PreferenceDataset`.

## 📞 Support

For issues or questions, open an issue on the repository or contact the maintainers.

