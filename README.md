# Algorithmic Memory Decay (AMD) — Simulation Guide

> **Paper**: "Predicting When Personalization Becomes Manipulation: A Behavioral Threshold Model for Social Media Algorithms"
>
> **Authors**: Vaibhav Kushwaha, Dr. Ruchika Gupta, Agam Sharma
>
> **Institution**: Department of Computer Science & Engineering, Chandigarh University, Mohali, India

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start (Run on Your Laptop)](#quick-start)
4. [Core Algorithm](#core-algorithm)
5. [Simulation Test Cases](#simulation-test-cases)
6. [Interactive Web Dashboard](#interactive-web-dashboard)
7. [Mathematical Formulation](#mathematical-formulation)
8. [Code Reference](#code-reference)
9. [Screenshots Gallery](#screenshots-gallery)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Modern social media recommender systems store vast histories of user behavior to maximize personalization. While this improves relevance, it creates serious problems:

- **Privacy threats**: Long-term behavioral profiling
- **Loss of autonomy**: Users trapped in filter bubbles
- **Algorithmic manipulation**: Past behavior disproportionately controls future exposure

**Algorithmic Memory Decay (AMD)** introduces **forgetting as a deliberate algorithmic mechanism** — not just a regulatory compliance measure. The **Temporal Ethical Memory Decay (TEMD)** algorithm and its extension **Entropy-Guided Adaptive Memory Decay (EG-AMD)** dynamically reduce the influence of older behavioral data while maintaining short-term relevance.

### Key Innovation

Instead of assuming "more data = better personalization," AMD asks:

> *Should a click from 3 years ago have the same influence as a click from yesterday?*

The answer is **no** — and AMD makes this principle mathematically rigorous.

---

## Project Structure

```
algorithmic-memory-decay/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── core/
│   └── temd.py                        # Core TEMD/EG-AMD algorithm implementation
│                                      # - TemporalEthicalMemoryDecay class
│                                      # - All 7 algorithm steps
│                                      # - State save/load utilities
│
├── simulations/
│   ├── test_cases.py                  # 4 simulation test cases from the paper
│   │                                    # - Simulation 1: Accuracy under decay
│   │                                    # - Simulation 2: Privacy-engagement trade-off
│   │                                    # - Simulation 3: Comparative analysis
│   │                                    # - Simulation 4: Entropy & dynamics
│   └── run_all_simulations.py         # Main runner (executes all simulations)
│
├── screenshots/                       # Generated simulation visualizations
│   ├── simulation_1_accuracy_3d.png
│   ├── simulation_1_accuracy_2d.png
│   ├── simulation_2_privacy_engagement.png
│   ├── simulation_2_contours.png
│   ├── simulation_3_comparative.png
│   ├── simulation_3_table.png
│   └── simulation_4_dynamics.png
│
└── webapp/                            # Interactive React dashboard
    ├── src/pages/Home.tsx             # Dashboard UI
    └── (built and deployed)
```

---

## Quick Start

### Step 1: Clone or Download

Extract this project to any folder on your laptop.

### Step 2: Install Dependencies

You need **Python 3.8+** installed.

```bash
# Navigate to project folder
cd algorithmic-memory-decay

# Create virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- `numpy` — Numerical computations
- `matplotlib` — Visualization and chart generation
- `scipy` — Signal processing (optional smoothing)

### Step 3: Run the Core Algorithm Demo

```bash
python core/temd.py
```

Expected output:
```
============================================================
TEMD Algorithm - Basic Demo
============================================================

Generated 10 users, 20 items, 100 interactions

User 0 has 12 interactions
Current decay coefficient: 0.1000

Top 5 Recommendations:
--------------------------------------------------------------------------------
1. Item 3 (news)
   Score: 0.8234 | Relevance: 0.4521 | Popularity: 0.2341
   Privacy Penalty: 0.1200 | Autonomy Bonus: 0.2573
...

Entropy: 0.7532
Exposure Risk: 0.4123
Updated Decay: 0.1089

============================================================
Demo complete!
============================================================
```

### Step 4: Run All Simulations

```bash
python simulations/run_all_simulations.py
```

This will:
1. Run all 4 test cases
2. Generate 7 visualization images
3. Save them to `screenshots/`

**Runtime**: ~30-60 seconds on a standard laptop.

### Step 5: View the Web Dashboard (Optional)

The interactive dashboard is deployed at:

**[https://42mn7bjvu5upi.kimi.show](https://42mn7bjvu5upi.kimi.show)**

Or run locally:
```bash
cd webapp
npm install
npm run dev
```

---

## Core Algorithm

The `core/temd.py` file implements the complete **Temporal Ethical Memory Decay (TEMD)** algorithm with these key classes:

### `UserInteraction`
Represents a single user-content interaction:
```python
interaction = UserInteraction(
    item_id=42,
    timestamp=24.5,    # Hours since start
    weight=2.0           # 1=click, 2=like, 3=share
)
```

### `ContentItem`
Represents content in the catalog:
```python
item = ContentItem(
    item_id=42,
    features=np.array([0.1, 0.3, -0.2, ...]),  # Feature vector
    popularity=0.35,
    category="tech"
)
```

### `TEMDConfig`
Algorithm hyperparameters:
```python
config = TEMDConfig(
    decay_coefficient=0.15,      # λ: base forgetting rate
    entropy_regularizer=0.5,     # β: autonomy bonus weight
    exposure_risk_weight=0.3,    # γ: privacy penalty weight
    learning_rate=0.01,          # α: adaptive learning rate
    min_decay=0.01,              # λ minimum bound
    max_decay=1.0,               # λ maximum bound
    target_entropy=0.5,          # target diversity
    cold_start_threshold=5       # min interactions for personalization
)
```

### `TemporalEthicalMemoryDecay`

Main algorithm class with these methods:

| Method | Description |
|--------|-------------|
| `add_interaction()` | Record new user interaction |
| `decay_kernel(t_delta, lambda)` | Compute `exp(-λ·Δt)` |
| `compute_decayed_preferences(t)` | Apply time-decay to all history |
| `compute_normalized_weights()` | Convert to probability distribution |
| `compute_entropy()` | Measure behavioral diversity |
| `compute_exposure_risk()` | Calculate profiling danger |
| `update_decay_coefficient()` | Adapt λ based on risk/entropy |
| `rank_items(candidates, t)` | Generate ethical recommendations |
| `save_state() / load_state()` | Persist/restore algorithm state |

### Basic Usage Example

```python
from core.temd import TemporalEthicalMemoryDecay, TEMDConfig, UserInteraction, ContentItem
import numpy as np

# Configure algorithm
config = TEMDConfig(decay_coefficient=0.2, entropy_regularizer=0.6)
user = TemporalEthicalMemoryDecay(config)

# Simulate user interactions
for t in range(20):
    interaction = UserInteraction(
        item_id=np.random.randint(0, 50),
        timestamp=float(t),
        weight=1.0
    )
    user.add_interaction(interaction)

# Rank content items
items = [ContentItem(i, np.random.randn(10)) for i in range(50)]
ranked = user.rank_items(items, current_time=25.0, return_scores=True)

# Print top 3
for item, score, details in ranked[:3]:
    print(f"Item {item.item_id}: score={score:.3f}, "
          f"entropy={details['entropy']:.3f}, "
          f"risk={details['exposure_risk']:.3f}")
```

---

## Simulation Test Cases

### Test Case 1: Recommendation Accuracy under Memory Decay

**Question**: Does intentional forgetting hurt recommendation quality?

**Method**: Simulate 30 users with drifting preferences over 40 time steps. Test 20 decay rates from 0.01 to 0.5.

**Result** (see `screenshots/simulation_1_accuracy_3d.png`):
- Accuracy grows initially as the system learns short-term preferences
- Higher decay reduces old-data influence but accuracy **plateaus at sustainable levels**
- TEMD effectively prioritizes recent, contextually relevant behavior

**Key Finding**: *Intentional forgetting does not sacrifice accuracy — it minimizes long-term dependency.*

---

### Test Case 2: Privacy–Engagement Trade-off

**Question**: Can we maintain engagement while reducing privacy exposure?

**Method**: Simulate 50 users across varying decay rates (0.01–0.6) and engagement levels (0.1–1.0).

**Result** (see `screenshots/simulation_2_privacy_engagement.png`):
- Privacy exposure **decreases significantly** with higher decay rates
- Engagement stays **moderate-to-high** even with aggressive forgetting
- The non-linear surface shows TEMD successfully **decouples engagement from data retention**

**Key Finding**: *Personalization without privacy sacrifice is possible via adaptive memory decay.*

---

### Test Case 3: Comparative Performance Analysis

**Question**: How does EG-AMD compare to existing personalization algorithms?

**Baselines**:
- Fu & Sun [1]: Adaptation-focused, persistent memory
- Yang & Xu [6]: Search personalization, persistent memory
- Lee & Johar [12]: Engagement-driven, persistent memory
- **EG-AMD (Proposed)**: Adaptive decay

**Result** (see `screenshots/simulation_3_comparative.png`):

| Algorithm | Accuracy | Privacy | Autonomy | Memory Handling |
|-----------|----------|---------|----------|-----------------|
| Fu & Sun | 0.68 | 0.00 | 0.91 | Persistent |
| Yang & Xu | 0.72 | 0.00 | 0.91 | Persistent |
| Lee & Johar | 0.68 | 0.00 | 0.91 | Persistent |
| **EG-AMD** | **0.79** | **0.01** | **0.91** | **Adaptive Decay** |

**Key Finding**: *EG-AMD achieves the best balance across all three ethical-performance dimensions.*

---

### Test Case 4: Behavioral Entropy and System Dynamics

**Question**: How do entropy, privacy risk, and decay coefficient evolve in real-time?

**Method**: Simulate 30 users with bursty interaction patterns over 50 time steps. Test 3 different configurations.

**Result** (see `screenshots/simulation_4_dynamics.png`):
- **Entropy** (autonomy): Stabilizes at high levels (~0.85–0.95) indicating diverse exposure
- **Exposure Risk** (privacy): Shows periodic dips during quiet periods, confirming adaptive response
- **Decay Coefficient**: Remains stable when properly configured, confirming system stability

**Key Finding**: *The adaptive feedback loop maintains ethical equilibrium without runaway behavior.*

---

## Mathematical Formulation

### Step 1: Temporal Interaction Modeling

User interaction history:

```
H = {(item₁, t₁, w₁), (item₂, t₂, w₂), ..., (itemₙ, tₙ, wₙ)}
```

Where:
- `itemᵢ` = content identifier
- `tᵢ` = timestamp
- `wᵢ` = interaction weight (click=1, like=2, share=3)

### Step 2: Exponential Decay Kernel

```
K(tᵢ, t) = exp(-λ · (t - tᵢ))
```

Where `λ` is the decay coefficient. Older interactions get exponentially smaller weights.

### Step 3: Decayed Preference Vector

```
p_decayed(t) = Σᵢ [ wᵢ · K(tᵢ, t) · e_itemᵢ ]
```

Where `e_itemᵢ` is the item's embedding vector.

### Step 4: Normalized Preference Weights

```
ŵᵢ = p_decayedᵢ / Σⱼ(p_decayedⱼ)
```

Converts raw weights to a probability distribution.

### Step 5: Behavioral Entropy (Autonomy Metric)

```
H = -Σᵢ [ ŵᵢ · log₂(ŵᵢ) ] / log₂(n)
```

Normalized to [0, 1]:
- **H ≈ 0**: Over-personalization (all weight on few items)
- **H ≈ 1**: Maximum diversity (uniform distribution)

### Step 6: Exposure Risk (Privacy Metric)

```
R = ||p_decayed||₂ / (√n + ε)
```

- **High R**: Concentrated preferences = easy profiling
- **Low R**: Distributed preferences = privacy-preserving

### Step 7: Entropy-Regularized Ranking Objective

Standard ranking maximizes relevance:
```
S_standard(item) = similarity(user, item)
```

TEMD extends this with ethical terms:
```
S_TEMD(item) = similarity(user, item) + popularity(item) - γ·R + β·H
```

Where:
- `similarity`: Personal relevance (cosine similarity)
- `popularity`: Global popularity (cold-start handling)
- `-γ·R`: Privacy penalty (exposure risk)
- `+β·H`: Autonomy bonus (entropy regularization)

### Step 8: Adaptive Decay Update

```
λₜ₊₁ = CLIP(λₜ + α · (Rₜ - (1 - Hₜ)), λ_min, λ_max)
```

Logic:
- If **R > (1-H)** → Profiling is high → **Increase λ** (forget more)
- If **H > (1-R)** → Diversity is high → **Decrease λ** (remember more)

---

## Code Reference

### Running Individual Simulations

```python
from simulations.test_cases import (
    simulation_1_accuracy_under_decay,
    simulation_2_privacy_engagement_tradeoff,
    simulation_3_comparative_analysis,
    simulation_4_entropy_and_dynamics
)

# Run just one simulation
result = simulation_1_accuracy_under_decay(
    n_users=50,
    n_items=30,
    time_steps=40
)

# Access results
print(result['figure_3d'])   # Path to saved PNG
print(result['accuracy_surface'])  # Raw data array
```

### Customizing Parameters

```python
from core.temd import TEMDConfig, TemporalEthicalMemoryDecay

# High-privacy configuration (aggressive forgetting)
high_privacy = TEMDConfig(
    decay_coefficient=0.4,
    entropy_regularizer=0.8,
    exposure_risk_weight=0.6
)

# High-engagement configuration (moderate forgetting)
high_engagement = TEMDConfig(
    decay_coefficient=0.05,
    entropy_regularizer=0.2,
    exposure_risk_weight=0.1
)

# Balanced configuration
balanced = TEMDConfig(
    decay_coefficient=0.15,
    entropy_regularizer=0.5,
    exposure_risk_weight=0.3
)
```

---

## Screenshots Gallery

All generated visualizations are saved in `screenshots/`:

| Screenshot | Description | Paper Figure |
|------------|-------------|--------------|
| `simulation_1_accuracy_3d.png` | 3D topography: Time × Decay → Accuracy | Figure 2 |
| `simulation_1_accuracy_2d.png` | 2D line plot: Accuracy trajectories | Supplementary |
| `simulation_2_privacy_engagement.png` | 3D surfaces: Privacy & Engagement | Figure 3 |
| `simulation_2_contours.png` | 2D contour: Privacy-Engagement trade-off | Supplementary |
| `simulation_3_comparative.png` | Bar chart & time series comparison | Figure 4 |
| `simulation_3_table.png` | Performance comparison table | Table in Section 5 |
| `simulation_4_dynamics.png` | Entropy, Risk, Decay over time | New analysis |

---

## Troubleshooting

### Issue: "Module not found" when running simulations

**Solution**: Make sure you're in the project root and Python can find the `core` module:

```bash
cd algorithmic-memory-decay
python -m simulations.run_all_simulations
```

Or set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/algorithmic-memory-decay:$PYTHONPATH
python simulations/run_all_simulations.py
```

### Issue: Matplotlib backend error

**Solution**: If running on a headless server or WSL, set the backend:

```bash
export MPLBACKEND=Agg
python simulations/run_all_simulations.py
```

### Issue: 3D plots not rendering

**Solution**: Make sure you have `matplotlib` with 3D support:

```bash
pip install --upgrade matplotlib
```

### Issue: Slow simulation performance

**Solution**: Reduce simulation parameters:

```python
simulation_1_accuracy_under_decay(
    n_users=10,      # Reduce from 50
    n_items=15,      # Reduce from 30
    time_steps=20    # Reduce from 40
)
```

---

## Citation

If you use this code or algorithm in your research, please cite:

```bibtex
@article{kushwaha2025amd,
  title={Predicting When Personalization Becomes Manipulation: 
         A Behavioral Threshold Model for Social Media Algorithms},
  author={Kushwaha, Vaibhav and Gupta, Ruchika and Sharma, Agam and Kumar, Ujjwal},
  institution={Chandigarh University, Mohali, India},
  year={2025}
}
```

---

## License

This simulation code is provided for academic and educational purposes.

---

**Questions?** Contact the authors:
- Vaibhav Kushwaha: professorauggie@gmail.com
- Dr. Ruchika Gupta: rgupt009@gmail.com

**Dashboard**: [https://42mn7bjvu5upi.kimi.show](https://42mn7bjvu5upi.kimi.show)
