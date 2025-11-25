# Personalized RLHF for Protein Design

This module provides infrastructure for personalized reinforcement learning from human feedback (RLHF) applied to protein design tasks. Instead of training on direct reward labels, the framework learns from **synthetic pairwise preferences** that reflect different user priorities and trade-offs.

## 🎯 Key Concepts

### Synthetic Users
Virtual users with specific property preferences (e.g., "PotencyMaximizer" prioritizes activity over toxicity).

### Pairwise Preferences
Training data in the form: "I prefer sequence A over sequence B" based on user-specific criteria.

### Preference Reward Models
Neural networks trained using Bradley-Terry loss to predict rewards from preferences.

### Personalized RL
RL training loops conditioned on user-specific reward models for customized design.

## 📁 Module Structure

```
personalization/
├── __init__.py                    # Module exports
├── synthetic_users.py             # User definitions and property computation
├── preference_generation.py       # Pairwise preference generation
├── preference_reward_model.py     # Bradley-Terry reward models
└── personalized_trainer.py        # Base classes for personalized RL
```

## 🚀 Quick Start

### 1. Define Synthetic Users

```python
from personalization import SyntheticUser

# Create a user who prioritizes activity over safety
user = SyntheticUser(
    name="PotencyMaximizer",
    weights={
        "activity_score": +1.0,
        "toxicity_score": 0.0,
        "stability": +0.3,
    },
    description="Maximizes antimicrobial activity"
)
```

### 2. Generate Preferences

```python
from personalization import generate_pairwise_preferences
import pandas as pd

# Your sequences with computed properties
df = pd.DataFrame({
    "sequence": ["GIGKFLHSAK", "KKLLPIVKKK", ...],
    "activity_score": [0.9, 0.8, ...],
    "toxicity_score": [0.6, 0.7, ...],
    "stability": [0.5, 0.4, ...],
})

# Generate 5000 preference pairs
prefs_df = generate_pairwise_preferences(
    df=df,
    user=user,
    num_pairs=5000,
    noise_flip_prob=0.05,  # 5% label noise
)
```

### 3. Train Preference Reward Model

```python
from personalization import PreferenceRewardModel, train_preference_reward_model
from personalization import PreferenceDataset
from torch.utils.data import DataLoader
import numpy as np

# Build feature matrix (e.g., ESM embeddings + properties)
features = np.array([...])  # Shape: (N, feature_dim)

# Create dataset
dataset = PreferenceDataset(features=features, preferences=prefs_df)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Create and train model
reward_model = PreferenceRewardModel(
    input_dim=features.shape[1],
    hidden_dims=[256, 128, 64],
    dropout=0.1,
)

trained_model, loss_history = train_preference_reward_model(
    model=reward_model,
    dataloader=dataloader,
    num_epochs=10,
    lr=1e-3,
    device="cuda",
)
```

### 4. Use in RL Training

```python
from personalization import PersonalizedRewardWrapper

# Wrap your existing reward function
personalized_reward_fn = PersonalizedRewardWrapper(
    base_reward_fn=my_classifier_reward,
    preference_reward_model=trained_model,
    feature_extractor=my_feature_extractor,
    blend_weight=0.5,  # 50% base, 50% personalized
)

# Use in your RL loop (PPO, GRPO, DPO, etc.)
rewards, mask = personalized_reward_fn(sequences)
```

## 🔬 Task-Specific Integration

### Antimicrobial Peptide (AMP) Design

```bash
# Train personalized AMP reward models
python amp_design/personalized_grpo.py \
    --classifier-checkpoint path/to/amp_classifier.pt \
    --output-dir personalized_rewards/amp \
    --num-pairs 5000
```

See `amp_design/personalized_grpo.py` for full integration.

### Antibody Mutation

```bash
# Train personalized antibody reward models
python antibody_mutation/personalized_mutation.py \
    --data-file data/antibodies.csv \
    --output-dir personalized_rewards/antibody \
    --num-pairs 3000
```

See `antibody_mutation/personalized_mutation.py` for full integration.

### Kinase Mutation

```bash
# Train personalized kinase reward models
python kinase_mutation/personalized_kinase.py \
    --data-file data/PhoQ.csv \
    --output-dir personalized_rewards/kinase \
    --num-pairs 2000
```

See `kinase_mutation/personalized_kinase.py` for full integration.

## 🎨 Pre-defined Synthetic Users

### For AMP Design

- **PotencyMaximizer**: Prioritizes activity above all
- **SafetyFirst**: Minimizes toxicity
- **BalancedDesigner**: Balanced multi-objective approach
- **ShortPeptideFan**: Prefers shorter sequences
- **StabilityFocused**: Prioritizes stability

### For Antibody Mutation

- **AffinityMaximizer**: Maximizes binding affinity
- **ConservativeMutator**: Fewer mutations with good properties
- **DevelopabilityFocused**: Optimizes for clinical development
- **StabilityEnhancer**: Focuses on improving stability

### For Kinase Mutation

- **ActivityBooster**: Maximizes kinase activity/fitness
- **ConservedEngineer**: Prefers conservative mutations
- **BalancedOptimizer**: Balanced fitness and stability

## 📊 Multi-User Training

Train models for multiple users simultaneously:

```python
from personalization import generate_multi_user_preferences
from personalization.synthetic_users import define_synthetic_users_amp

users = define_synthetic_users_amp()

combined_prefs, user_index = generate_multi_user_preferences(
    df=df,
    users=users,
    num_pairs_per_user=3000,
)

# Train a single model conditioned on user embeddings
reward_model = PreferenceRewardModel(
    input_dim=feature_dim,
    num_users=len(users),
    user_embed_dim=32,
    use_user_conditioning=True,
)
```

## 🔧 Advanced Features

### Ensemble Reward Models

```python
from personalization.preference_reward_model import EnsembleRewardModel

models = [trained_model_1, trained_model_2, trained_model_3]
ensemble = EnsembleRewardModel(models)

# Get predictions with uncertainty
mean_rewards, std_rewards = ensemble(features, return_uncertainty=True)
```

### Custom Property Functions

```python
from personalization import compute_property_scores

property_functions = {
    "my_custom_property": lambda seq: my_computation(seq),
    "stability": my_stability_predictor,
}

df = compute_property_scores(
    sequences=sequences,
    property_functions=property_functions,
)
```

## 📈 Evaluation

Evaluate your trained reward models:

```python
from personalization.preference_reward_model import evaluate_reward_model

metrics = evaluate_reward_model(
    model=trained_model,
    dataloader=test_dataloader,
    device="cuda",
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Loss: {metrics['loss']:.4f}")
```

## 🔄 Inference at Test Time

Once trained, you can condition generation on different users:

```python
# Train multiple RL policies, one per user
for user in users:
    reward_fn = create_user_conditioned_reward_fn(
        preference_reward_model=multi_user_model,
        feature_extractor=feature_extractor,
        user_id=user_id,
    )
    
    # Train RL policy with this reward function
    train_rl_policy(reward_fn=reward_fn, user_name=user.name)
```

Or use a single policy with user-specific rewards at inference:

```python
# Generate for "SafetyFirst" user
sequences_safe = generate_with_user(policy, user_id=1)

# Generate for "PotencyMaximizer" user
sequences_potent = generate_with_user(policy, user_id=0)
```

## 📝 Citation

If you use this personalization framework, please cite:

```bibtex
@article{yourpaper2025,
  title={Personalized RLHF for Protein Design},
  author={Your Name et al.},
  journal={Your Journal},
  year={2025}
}
```

## 🤝 Contributing

To add support for new tasks:

1. Define task-specific synthetic users in `synthetic_users.py`
2. Implement feature extraction for your sequences
3. Create a personalized trainer following the `PersonalizedRLTrainer` interface
4. Add integration examples

## 📚 Further Reading

- **Bradley-Terry Model**: [Wikipedia](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- **RLHF**: [OpenAI's InstructGPT paper](https://arxiv.org/abs/2203.02155)
- **Preference Learning**: Survey on learning from preferences in ML

## ❓ Troubleshooting

### Issue: Preference model has low accuracy
- **Solution**: Increase number of preference pairs, reduce noise_flip_prob, or add more informative features

### Issue: RL training is unstable with personalized rewards
- **Solution**: Start with lower blend_weight (e.g., 0.2) to rely more on base reward initially

### Issue: Different users produce similar sequences
- **Solution**: Increase diversity in user weight definitions or use stronger regularization in RL

## 📄 License

This module is part of the RLHF-PLM project. See main repository LICENSE for details.

