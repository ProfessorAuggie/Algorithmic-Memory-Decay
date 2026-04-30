"""
Run All Simulations
===================

Main entry point to execute all TEMD simulations and generate outputs.

Usage:
    python run_all_simulations.py

This will:
1. Run Simulation 1: Recommendation Accuracy under Memory Decay
2. Run Simulation 2: Privacy-Engagement Trade-off
3. Run Simulation 3: Comparative Analysis
4. Run Simulation 4: Entropy and System Dynamics
5. Save all figures to ../screenshots/
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.test_cases import (
    simulation_1_accuracy_under_decay,
    simulation_2_privacy_engagement_tradeoff,
    simulation_3_comparative_analysis,
    simulation_4_entropy_and_dynamics
)
import json


def main():
    print("=" * 70)
    print("  Algorithmic Memory Decay (AMD) - Simulation Suite")
    print("  Paper: 'Predicting When Personalization Becomes Manipulation'")
    print("  Authors: Kushwaha, Gupta, Sharma, Kumar")
    print("  Institution: Chandigarh University")
    print("=" * 70)
    print()
    
    results = {}
    
    # Simulation 1
    print("\n[1/4] Recommendation Accuracy under Temporal Memory Decay")
    print("-" * 50)
    results['simulation_1'] = simulation_1_accuracy_under_decay(
        n_users=30,
        n_items=20,
        time_steps=40,
        random_seed=42
    )
    
    # Simulation 2
    print("\n[2/4] Privacy vs Engagement Trade-off")
    print("-" * 50)
    results['simulation_2'] = simulation_2_privacy_engagement_tradeoff(
        n_users=50,
        n_items=30,
        random_seed=42
    )
    
    # Simulation 3
    print("\n[3/4] Comparative Performance Analysis")
    print("-" * 50)
    results['simulation_3'] = simulation_3_comparative_analysis(
        n_users=60,
        n_items=40,
        time_steps=25,
        random_seed=42
    )
    
    # Simulation 4
    print("\n[4/4] Entropy and System Dynamics")
    print("-" * 50)
    results['simulation_4'] = simulation_4_entropy_and_dynamics(
        n_users=30,
        time_steps=50
    )
    
    # Save summary
    screenshots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
    summary = {
        'simulations': {
            name: {
                'figures': [v for k, v in data.items() if 'figure' in k or 'table' in k],
                'description': data.get('simulation', name)
            }
            for name, data in results.items()
        },
        'output_directory': screenshots_dir
    }
    
    summary_path = os.path.join(screenshots_dir, 'simulation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  ALL SIMULATIONS COMPLETE!")
    print("=" * 70)
    print(f"\n  Output files:")
    for sim_name, data in results.items():
        print(f"\n  {sim_name}:")
        for key, val in data.items():
            if 'figure' in key or 'table' in key:
                print(f"    - {val}")
    
    print(f"\n  Summary JSON: {summary_path}")
    print("\n" + "=" * 70)
    
    return results


if __name__ == '__main__':
    main()
