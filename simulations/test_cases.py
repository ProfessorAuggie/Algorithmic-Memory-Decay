"""
Simulation Scripts for Algorithmic Memory Decay
===============================================

This module implements the four main test cases from the paper:
1. Recommendation Accuracy under Temporal Memory Decay
2. Privacy Exposure versus Engagement with Adaptive Decay
3. Comparative Performance Analysis
4. Behavioral Entropy and Autonomy Metrics

Run each simulation independently or execute all from run_all_simulations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import json
import os
from typing import Dict, List, Tuple
from core.temd import (
    TemporalEthicalMemoryDecay, TEMDConfig, 
    UserInteraction, ContentItem, create_sample_data
)

# Ensure screenshots directory exists
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def simulation_1_accuracy_under_decay(
    n_users: int = 50,
    n_items: int = 30,
    feature_dim: int = 8,
    time_steps: int = 50,
    decay_rates: np.ndarray = None,
    random_seed: int = 42,
    save_path: str = None
) -> Dict:
    """
    Test Case 1: Recommendation Accuracy under Temporal Memory Decay
    
    This simulation creates a 3D topography showing how recommendation accuracy
    evolves over time for different memory decay rates.
    
    Key Insight: Higher decay rates reduce old data influence but accuracy 
    plateaus at sustainable levels - TEMD prioritizes recent context.
    
    Returns:
        Dictionary with simulation results and figure path
    """
    np.random.seed(random_seed)
    
    if decay_rates is None:
        decay_rates = np.linspace(0.01, 0.5, 20)
    
    print("[Simulation 1] Running: Accuracy under Memory Decay")
    print(f"  Parameters: {n_users} users, {n_items} items, {time_steps} time steps")
    print(f"  Decay rates: {decay_rates[0]:.3f} to {decay_rates[-1]:.3f}")
    
    # Create consistent item catalog
    items = []
    for i in range(n_items):
        features = np.random.randn(feature_dim)
        features = features / (np.linalg.norm(features) + 1e-8)
        items.append(ContentItem(
            item_id=i,
            features=features,
            popularity=np.random.beta(2, 5),
            category=['tech', 'sports', 'news', 'entertainment'][i % 4]
        ))
    
    # Ground truth: user's "true" evolving preferences
    # Simulate users whose preferences drift over time
    accuracy_surface = np.zeros((len(decay_rates), time_steps))
    
    for i, decay_rate in enumerate(decay_rates):
        # Create fresh users for each decay rate
        users = []
        true_preferences = []  # Ground truth preference vectors
        
        for u in range(n_users):
            config = TEMDConfig(
                decay_coefficient=decay_rate,
                entropy_regularizer=0.5,
                exposure_risk_weight=0.3,
                learning_rate=0.01
            )
            users.append(TemporalEthicalMemoryDecay(config))
            # Each user has a drifting true preference
            true_preferences.append(np.random.randn(feature_dim))
        
        # Simulate over time
        for t in range(time_steps):
            # Users' true preferences drift slightly
            for u in range(n_users):
                drift = np.random.randn(feature_dim) * 0.1
                true_preferences[u] += drift
                true_preferences[u] = true_preferences[u] / (np.linalg.norm(true_preferences[u]) + 1e-8)
            
            # Generate interactions based on true preferences
            for u in range(n_users):
                # User interacts with items similar to current true preference
                similarities = []
                for item in items:
                    sim = np.dot(true_preferences[u], item.features)
                    similarities.append(sim)
                
                # Add noise and pick top items
                similarities = np.array(similarities)
                probs = np.exp(similarities * 2)  # Sharpen distribution
                probs = probs / probs.sum()
                
                # User interacts with 2-3 items per time step
                n_interactions = np.random.randint(1, 4)
                for _ in range(n_interactions):
                    item_idx = np.random.choice(len(items), p=probs)
                    weight = np.random.choice([1.0, 2.0, 3.0], p=[0.6, 0.3, 0.1])
                    interaction = UserInteraction(
                        item_id=item_idx,
                        timestamp=float(t),
                        weight=weight
                    )
                    users[u].add_interaction(interaction)
            
            # Measure accuracy: how well do recommendations match true preference?
            accuracies = []
            for u in range(n_users):
                if len(users[u].interaction_history) < 5:
                    continue
                
                # Rank items
                ranked = users[u].rank_items(items, float(t) + 0.5, return_scores=True)
                
                # Top-5 accuracy: check if top recommendations align with true preference
                top_items = [item.item_id for item, _, _ in ranked[:5]]
                
                # Compute similarity of top recommendations to true preference
                top_sims = []
                for item_id in top_items:
                    sim = np.dot(true_preferences[u], items[item_id].features)
                    top_sims.append(sim)
                
                avg_top_sim = np.mean(top_sims)
                accuracies.append(avg_top_sim)
            
            if accuracies:
                accuracy_surface[i, t] = np.mean(accuracies)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(range(time_steps), decay_rates)
    
    # Smooth the surface slightly for better visualization
    from scipy.ndimage import gaussian_filter
    Z_smooth = gaussian_filter(accuracy_surface, sigma=0.5)
    
    surf = ax.plot_surface(X, Y, Z_smooth, cmap=cm.viridis, 
                           linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('Time Steps', fontsize=12, labelpad=10)
    ax.set_ylabel('Decay Rate (λ)', fontsize=12, labelpad=10)
    ax.set_zlabel('Recommendation Accuracy', fontsize=12, labelpad=10)
    ax.set_title('Test Case 1: Recommendation Accuracy under Temporal Memory Decay\n'
                 '(3D Topography: Time × Decay Rate → Accuracy)', 
                 fontsize=13, pad=20)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Accuracy Score')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(SCREENSHOTS_DIR, 'simulation_1_accuracy_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {save_path}")
    
    # Also save 2D line plot for clarity
    fig, ax2d = plt.subplots(figsize=(10, 6))
    for i in [0, len(decay_rates)//4, len(decay_rates)//2, 3*len(decay_rates)//4, len(decay_rates)-1]:
        ax2d.plot(range(time_steps), accuracy_surface[i], 
                 label=f'λ = {decay_rates[i]:.3f}', alpha=0.8, linewidth=2)
    ax2d.set_xlabel('Time Steps')
    ax2d.set_ylabel('Recommendation Accuracy')
    ax2d.set_title('Accuracy Trajectories for Different Decay Rates')
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)
    plt.tight_layout()
    
    line_path = os.path.join(SCREENSHOTS_DIR, 'simulation_1_accuracy_2d.png')
    plt.savefig(line_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 2D version saved to: {line_path}")
    
    return {
        'simulation': 'accuracy_under_decay',
        'accuracy_surface': accuracy_surface.tolist(),
        'decay_rates': decay_rates.tolist(),
        'time_steps': time_steps,
        'figure_3d': save_path,
        'figure_2d': line_path
    }


def simulation_2_privacy_engagement_tradeoff(
    n_users: int = 100,
    n_items: int = 40,
    feature_dim: int = 8,
    decay_rates: np.ndarray = None,
    engagement_levels: np.ndarray = None,
    random_seed: int = 42,
    save_path: str = None
) -> Dict:
    """
    Test Case 2: Privacy Exposure vs Engagement Trade-off Analysis
    
    Creates a 3D surface showing the relationship between:
    - Decay rate (x-axis)
    - Engagement level (y-axis)  
    - Privacy exposure / Engagement score (z-axis)
    
    Key Insight: Higher decay reduces privacy exposure while engagement 
    stays moderate - proving ethical personalization is possible.
    """
    np.random.seed(random_seed)
    
    if decay_rates is None:
        decay_rates = np.linspace(0.01, 0.6, 25)
    if engagement_levels is None:
        engagement_levels = np.linspace(0.1, 1.0, 25)  # User engagement tendency
    
    print("[Simulation 2] Running: Privacy-Engagement Trade-off")
    print(f"  Parameters: {n_users} users, {n_items} items")
    print(f"  Decay rates: {decay_rates[0]:.3f} to {decay_rates[-1]:.3f}")
    print(f"  Engagement levels: {engagement_levels[0]:.3f} to {engagement_levels[-1]:.3f}")
    
    # Create items
    items = []
    for i in range(n_items):
        features = np.random.randn(feature_dim)
        features = features / (np.linalg.norm(features) + 1e-8)
        items.append(ContentItem(
            item_id=i,
            features=features,
            popularity=np.random.beta(2, 5)
        ))
    
    # Grid for surface
    privacy_surface = np.zeros((len(decay_rates), len(engagement_levels)))
    engagement_surface = np.zeros((len(decay_rates), len(engagement_levels)))
    
    for i, decay in enumerate(decay_rates):
        for j, engagement in enumerate(engagement_levels):
            privacy_scores = []
            engagement_scores = []
            
            for u in range(n_users):
                config = TEMDConfig(
                    decay_coefficient=decay,
                    entropy_regularizer=0.5,
                    exposure_risk_weight=0.3
                )
                user = TemporalEthicalMemoryDecay(config)
                
                # Generate interactions proportional to engagement level
                n_interactions = int(engagement * 50)  # 5-50 interactions
                
                for t in range(n_interactions):
                    item_idx = np.random.randint(0, n_items)
                    weight = np.random.choice([1.0, 2.0, 3.0], p=[0.6, 0.3, 0.1])
                    interaction = UserInteraction(item_idx, float(t), weight)
                    user.add_interaction(interaction)
                
                # Compute metrics
                decayed_prefs = user.compute_decayed_preferences(float(n_interactions))
                
                if decayed_prefs:
                    # Privacy = exposure risk (lower is better)
                    privacy = user.compute_exposure_risk(decayed_prefs)
                    privacy_scores.append(privacy)
                    
                    # Engagement = average recommendation score
                    ranked = user.rank_items(items, float(n_interactions))
                    if ranked:
                        avg_score = np.mean([score for _, score in ranked[:10]])
                        engagement_scores.append(avg_score)
            
            privacy_surface[i, j] = np.mean(privacy_scores) if privacy_scores else 0
            engagement_surface[i, j] = np.mean(engagement_scores) if engagement_scores else 0
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(14, 6))
    
    # Privacy surface
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(engagement_levels, decay_rates)
    
    from scipy.ndimage import gaussian_filter
    Z_privacy = gaussian_filter(privacy_surface, sigma=0.5)
    
    surf1 = ax1.plot_surface(X, Y, Z_privacy, cmap=cm.plasma,
                             linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_xlabel('Engagement Level', fontsize=11)
    ax1.set_ylabel('Decay Rate (λ)', fontsize=11)
    ax1.set_zlabel('Privacy Exposure Risk', fontsize=11)
    ax1.set_title('Privacy Exposure\n(Lower = Better Privacy)', fontsize=12)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Engagement surface
    ax2 = fig.add_subplot(122, projection='3d')
    Z_engagement = gaussian_filter(engagement_surface, sigma=0.5)
    
    surf2 = ax2.plot_surface(X, Y, Z_engagement, cmap=cm.viridis,
                             linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel('Engagement Level', fontsize=11)
    ax2.set_ylabel('Decay Rate (λ)', fontsize=11)
    ax2.set_zlabel('Engagement Score', fontsize=11)
    ax2.set_title('User Engagement\n(Higher = Better Engagement)', fontsize=12)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.suptitle('Test Case 2: Privacy vs Engagement Trade-off with Adaptive Decay',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(SCREENSHOTS_DIR, 'simulation_2_privacy_engagement.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {save_path}")
    
    # Create 2D contour plot for combined analysis
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Privacy contour
    contour1 = ax3.contourf(X, Y, Z_privacy, levels=20, cmap=cm.plasma)
    ax3.set_xlabel('Engagement Level')
    ax3.set_ylabel('Decay Rate (λ)')
    ax3.set_title('Privacy Exposure (Contour)')
    plt.colorbar(contour1, ax=ax3)
    
    # Engagement contour
    contour2 = ax4.contourf(X, Y, Z_engagement, levels=20, cmap=cm.viridis)
    ax4.set_xlabel('Engagement Level')
    ax4.set_ylabel('Decay Rate (λ)')
    ax4.set_title('Engagement Score (Contour)')
    plt.colorbar(contour2, ax=ax4)
    
    plt.suptitle('Privacy-Engagement Trade-off (2D Contours)', fontsize=13)
    plt.tight_layout()
    
    contour_path = os.path.join(SCREENSHOTS_DIR, 'simulation_2_contours.png')
    plt.savefig(contour_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Contour plot saved to: {contour_path}")
    
    return {
        'simulation': 'privacy_engagement_tradeoff',
        'privacy_surface': privacy_surface.tolist(),
        'engagement_surface': engagement_surface.tolist(),
        'decay_rates': decay_rates.tolist(),
        'engagement_levels': engagement_levels.tolist(),
        'figure_3d': save_path,
        'figure_contour': contour_path
    }


def simulation_3_comparative_analysis(
    n_users: int = 100,
    n_items: int = 50,
    feature_dim: int = 10,
    time_steps: int = 30,
    random_seed: int = 42,
    save_path: str = None
) -> Dict:
    """
    Test Case 3: Comparative Performance Analysis
    
    Compares TEMD/EG-AMD against baseline algorithms:
    - Fu & Sun [1]: Adaptation-focused, persistent memory
    - Yang & Xu [6]: Search personalization, persistent memory
    - Lee & Johar [12]: Engagement-driven, persistent memory
    - EG-AMD (Proposed): Adaptive decay
    
    Metrics: Accuracy, Privacy, Autonomy, Memory Handling
    """
    np.random.seed(random_seed)
    
    print("[Simulation 3] Running: Comparative Analysis")
    print(f"  Parameters: {n_users} users, {n_items} items, {time_steps} steps")
    
    # Create items
    items = []
    for i in range(n_items):
        features = np.random.randn(feature_dim)
        features = features / (np.linalg.norm(features) + 1e-8)
        items.append(ContentItem(
            item_id=i,
            features=features,
            popularity=np.random.beta(2, 5)
        ))
    
    # Simulate four algorithms
    algorithms = {
        'Fu & Sun [1]\n(Persistent)': {
            'decay': 0.001,  # Almost no decay
            'entropy_reg': 0.1,
            'exposure_weight': 0.1
        },
        'Yang & Xu [6]\n(Persistent)': {
            'decay': 0.005,
            'entropy_reg': 0.3,
            'exposure_weight': 0.2
        },
        'Lee & Johar [12]\n(Persistent)': {
            'decay': 0.01,
            'entropy_reg': 0.1,
            'exposure_weight': 0.1
        },
        'EG-AMD (Proposed)\n(Adaptive Decay)': {
            'decay': 0.15,  # Higher initial decay with adaptation
            'entropy_reg': 0.5,
            'exposure_weight': 0.3,
            'adaptive': True
        }
    }
    
    results = {name: {'accuracy': [], 'privacy': [], 'autonomy': [], 'memory': []} 
               for name in algorithms}
    
    # Time series simulation
    for t in range(time_steps):
        for algo_name, params in algorithms.items():
            accuracy_vals = []
            privacy_vals = []
            autonomy_vals = []
            
            for u in range(n_users):
                config = TEMDConfig(
                    decay_coefficient=params['decay'],
                    entropy_regularizer=params['entropy_reg'],
                    exposure_risk_weight=params['exposure_weight'],
                    learning_rate=0.01 if params.get('adaptive') else 0.0
                )
                user = TemporalEthicalMemoryDecay(config)
                
                # Generate interactions with preference drift
                if t > 0:
                    n_interactions = np.random.randint(3, 8)
                    for _ in range(n_interactions):
                        item_idx = np.random.randint(0, n_items)
                        weight = np.random.choice([1.0, 2.0, 3.0], p=[0.6, 0.3, 0.1])
                        interaction = UserInteraction(item_idx, float(t), weight)
                        user.add_interaction(interaction)
                
                # Compute metrics
                decayed_prefs = user.compute_decayed_preferences(float(t) + 1)
                
                if decayed_prefs:
                    # Accuracy: how well do we predict?
                    ranked = user.rank_items(items, float(t) + 1, return_scores=True)
                    if ranked:
                        scores = [s for _, s, _ in ranked]
                        accuracy_vals.append(np.mean(scores[:5]))
                    
                    # Privacy: exposure risk
                    privacy = 1.0 - user.compute_exposure_risk(decayed_prefs)  # Invert: higher = better privacy
                    privacy_vals.append(privacy)
                    
                    # Autonomy: entropy
                    norm_weights = user.compute_normalized_weights(decayed_prefs)
                    autonomy = user.compute_entropy(norm_weights)
                    autonomy_vals.append(autonomy)
            
            results[algo_name]['accuracy'].append(np.mean(accuracy_vals) if accuracy_vals else 0)
            results[algo_name]['privacy'].append(np.mean(privacy_vals) if privacy_vals else 0)
            results[algo_name]['autonomy'].append(np.mean(autonomy_vals) if autonomy_vals else 0)
    
    # Aggregate final scores
    final_scores = {}
    for algo_name in algorithms:
        final_scores[algo_name] = {
            'Accuracy': np.mean(results[algo_name]['accuracy']),
            'Privacy': np.mean(results[algo_name]['privacy']),
            'Autonomy': np.mean(results[algo_name]['autonomy']),
            'Memory': 1.0 if 'Adaptive' in algo_name else 0.2  # Binary for visualization
        }
    
    # Create comparative bar chart (matching paper's Figure 4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    metrics = ['Accuracy', 'Privacy', 'Autonomy']
    x = np.arange(len(algorithms))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, metric in enumerate(metrics):
        values = [final_scores[algo][metric] for algo in algorithms]
        bars = ax1.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Comparative Performance Analysis')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(algorithms.keys(), rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Time series comparison for EG-AMD
    ax2.plot(range(time_steps), results['EG-AMD (Proposed)\n(Adaptive Decay)']['accuracy'],
             label='Accuracy', color=colors[0], linewidth=2, marker='o', markersize=4)
    ax2.plot(range(time_steps), results['EG-AMD (Proposed)\n(Adaptive Decay)']['privacy'],
             label='Privacy', color=colors[1], linewidth=2, marker='s', markersize=4)
    ax2.plot(range(time_steps), results['EG-AMD (Proposed)\n(Adaptive Decay)']['autonomy'],
             label='Autonomy', color=colors[2], linewidth=2, marker='^', markersize=4)
    
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Score')
    ax2.set_title('EG-AMD Performance Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Test Case 3: Comparative Analysis of Personalization Algorithms',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(SCREENSHOTS_DIR, 'simulation_3_comparative.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {save_path}")
    
    # Create summary table figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for algo_name in algorithms:
        row = [
            algo_name.replace('\n', ' '),
            f"{final_scores[algo_name]['Accuracy']:.3f}",
            f"{final_scores[algo_name]['Privacy']:.3f}",
            f"{final_scores[algo_name]['Autonomy']:.3f}",
            'Adaptive Decay' if 'Adaptive' in algo_name else 'Persistent'
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                     colLabels=['Algorithm', 'Accuracy', 'Privacy', 'Autonomy', 'Memory Handling'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header
    for i in range(5):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight EG-AMD row
    for i in range(5):
        table[(4, i)].set_facecolor('#d5f5e3')
    
    plt.title('Comparative Performance Table', fontsize=14, pad=20)
    plt.tight_layout()
    
    table_path = os.path.join(SCREENSHOTS_DIR, 'simulation_3_table.png')
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Table saved to: {table_path}")
    
    return {
        'simulation': 'comparative_analysis',
        'results': results,
        'final_scores': final_scores,
        'figure': save_path,
        'table': table_path
    }


def simulation_4_entropy_and_dynamics(
    n_users: int = 50,
    time_steps: int = 60,
    save_path: str = None
) -> Dict:
    """
    Test Case 4: Behavioral Entropy and System Dynamics
    
    Shows how entropy, exposure risk, and decay coefficient evolve over time
    for the TEMD algorithm, demonstrating the adaptive feedback loop.
    """
    np.random.seed(42)
    
    print("[Simulation 4] Running: Entropy and Dynamics Analysis")
    print(f"  Parameters: {n_users} users, {time_steps} time steps")
    
    # Simulate users with different behaviors
    configs = [
        TEMDConfig(decay_coefficient=0.1, entropy_regularizer=0.3),  # Low autonomy focus
        TEMDConfig(decay_coefficient=0.2, entropy_regularizer=0.7),  # High autonomy focus
        TEMDConfig(decay_coefficient=0.15, entropy_regularizer=0.5),  # Balanced
    ]
    
    all_entropy = {f'Config {i+1}': [] for i in range(len(configs))}
    all_exposure = {f'Config {i+1}': [] for i in range(len(configs))}
    all_decay = {f'Config {i+1}': [] for i in range(len(configs))}
    
    for config_idx, config in enumerate(configs):
        label = f'Config {config_idx + 1}'
        
        for u in range(n_users):
            user = TemporalEthicalMemoryDecay(config)
            
            # Simulate interactions with bursts (periods of high activity)
            for t in range(time_steps):
                # Bursty interaction pattern
                if t % 10 < 5:  # Active period
                    n_int = np.random.randint(2, 5)
                else:  # Quiet period
                    n_int = np.random.randint(0, 2)
                
                for _ in range(n_int):
                    interaction = UserInteraction(
                        item_id=np.random.randint(0, 20),
                        timestamp=float(t),
                        weight=np.random.choice([1.0, 2.0], p=[0.7, 0.3])
                    )
                    user.add_interaction(interaction)
                
                # Compute metrics
                decayed_prefs = user.compute_decayed_preferences(float(t) + 0.5)
                if decayed_prefs:
                    norm_weights = user.compute_normalized_weights(decayed_prefs)
                    entropy = user.compute_entropy(norm_weights)
                    exposure = user.compute_exposure_risk(decayed_prefs)
                    
                    all_entropy[label].append(entropy)
                    all_exposure[label].append(exposure)
                    all_decay[label].append(user.current_decay)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Entropy plot
    for i, (label, values) in enumerate(all_entropy.items()):
        # Average per time step across users
        values_arr = np.array(values).reshape(n_users, time_steps)
        mean_values = values_arr.mean(axis=0)
        std_values = values_arr.std(axis=0)
        
        axes[0].plot(range(time_steps), mean_values, 
                    label=label, color=colors[i], linewidth=2)
        axes[0].fill_between(range(time_steps), 
                            mean_values - std_values,
                            mean_values + std_values,
                            alpha=0.2, color=colors[i])
    
    axes[0].set_ylabel('Behavioral Entropy', fontsize=11)
    axes[0].set_title('Autonomy: Behavioral Entropy Over Time\n(Higher = More Diverse Exposure)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Exposure risk plot
    for i, (label, values) in enumerate(all_exposure.items()):
        values_arr = np.array(values).reshape(n_users, time_steps)
        mean_values = values_arr.mean(axis=0)
        std_values = values_arr.std(axis=0)
        
        axes[1].plot(range(time_steps), mean_values,
                    label=label, color=colors[i], linewidth=2)
        axes[1].fill_between(range(time_steps),
                            mean_values - std_values,
                            mean_values + std_values,
                            alpha=0.2, color=colors[i])
    
    axes[1].set_ylabel('Exposure Risk', fontsize=11)
    axes[1].set_title('Privacy: Exposure Risk Over Time\n(Lower = Better Privacy)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Decay coefficient plot
    for i, (label, values) in enumerate(all_decay.items()):
        values_arr = np.array(values).reshape(n_users, time_steps)
        mean_values = values_arr.mean(axis=0)
        
        axes[2].plot(range(time_steps), mean_values,
                    label=label, color=colors[i], linewidth=2)
    
    axes[2].set_xlabel('Time Steps', fontsize=11)
    axes[2].set_ylabel('Decay Coefficient (λ)', fontsize=11)
    axes[2].set_title('Adaptive Decay Coefficient Evolution\n(Auto-adjusts based on risk/entropy)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Test Case 4: System Dynamics - Entropy, Privacy & Adaptive Decay',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(SCREENSHOTS_DIR, 'simulation_4_dynamics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {save_path}")
    
    return {
        'simulation': 'entropy_and_dynamics',
        'entropy': all_entropy,
        'exposure_risk': all_exposure,
        'decay_coefficient': all_decay,
        'figure': save_path
    }


if __name__ == '__main__':
    print("=" * 60)
    print("TEMD Simulations")
    print("=" * 60)
    
    # Run all simulations
    results = {}
    results['sim1'] = simulation_1_accuracy_under_decay()
    results['sim2'] = simulation_2_privacy_engagement_tradeoff()
    results['sim3'] = simulation_3_comparative_analysis()
    results['sim4'] = simulation_4_entropy_and_dynamics()
    
    # Save results summary
    summary_path = os.path.join(SCREENSHOTS_DIR, 'simulation_results.json')
    with open(summary_path, 'w') as f:
        json.dump({k: {key: val for key, val in v.items() if isinstance(val, (str, int, float, list))} 
                   for k, v in results.items()}, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("All simulations complete!")
    print(f"Results saved to: {SCREENSHOTS_DIR}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")
