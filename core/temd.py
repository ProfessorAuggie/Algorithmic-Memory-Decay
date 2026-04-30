"""
Algorithmic Memory Decay (AMD) - Core Implementation
====================================================

This module implements the Temporal Ethical Memory Decay (TEMD) algorithm
and the Entropy-Guided Adaptive Memory Decay (EG-AMD) framework as described in:

"Predicting When Personalization Becomes Manipulation: 
 A Behavioral Threshold Model for Social Media Algorithms"

Authors: Vaibhav Kushwaha, Dr. Ruchika Gupta, Agam Sharma, Ujjwal Kumar
Institution: Chandigarh University, Mohali, India
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import json


@dataclass
class UserInteraction:
    """Represents a single user interaction with content."""
    item_id: int
    timestamp: float  # Time in hours (or any consistent unit)
    weight: float = 1.0  # Interaction strength (click=1, like=2, share=3, etc.)
    
    def __repr__(self):
        return f"Interaction(item={self.item_id}, t={self.timestamp:.2f}, w={self.weight:.2f})"


@dataclass
class ContentItem:
    """Represents a content item in the system."""
    item_id: int
    features: np.ndarray  # Content feature vector
    popularity: float = 0.0  # Global popularity score
    category: str = "general"
    
    def __post_init__(self):
        if isinstance(self.features, list):
            self.features = np.array(self.features)


@dataclass
class TEMDConfig:
    """Configuration parameters for the TEMD algorithm."""
    decay_coefficient: float = 0.1  # Base decay rate (lambda)
    entropy_regularizer: float = 0.5  # Beta: entropy regularization weight
    exposure_risk_weight: float = 0.3  # Gamma: exposure risk penalty weight
    learning_rate: float = 0.01  # Alpha: adaptive learning rate for decay
    min_decay: float = 0.01  # Minimum decay coefficient
    max_decay: float = 1.0  # Maximum decay coefficient
    target_entropy: float = 0.5  # Target entropy for autonomy (0=uniform, 1=max)
    cold_start_threshold: int = 5  # Min interactions before personalization
    
    def to_dict(self):
        return {
            'decay_coefficient': self.decay_coefficient,
            'entropy_regularizer': self.entropy_regularizer,
            'exposure_risk_weight': self.exposure_risk_weight,
            'learning_rate': self.learning_rate,
            'min_decay': self.min_decay,
            'max_decay': self.max_decay,
            'target_entropy': self.target_entropy,
            'cold_start_threshold': self.cold_start_threshold
        }


class TemporalEthicalMemoryDecay:
    """
    Temporal Ethical Memory Decay (TEMD) Algorithm
    
    Implements controlled forgetting in recommender systems by:
    1. Applying exponential time-decay to historical interactions
    2. Measuring behavioral entropy for autonomy assessment
    3. Regularizing rankings with entropy and exposure risk
    4. Adaptively updating decay based on privacy-engagement trade-off
    """
    
    def __init__(self, config: Optional[TEMDConfig] = None):
        self.config = config or TEMDConfig()
        self.interaction_history: List[UserInteraction] = []
        self.current_decay: float = self.config.decay_coefficient
        self.exposure_risk_history: List[float] = []
        self.entropy_history: List[float] = []
        self.accuracy_history: List[float] = []
        
    # =========================================================================
    # STEP 1: Temporal Interaction Modeling
    # =========================================================================
    
    def add_interaction(self, interaction: UserInteraction) -> None:
        """Add a new user interaction to the history."""
        self.interaction_history.append(interaction)
        # Keep history sorted by timestamp
        self.interaction_history.sort(key=lambda x: x.timestamp)
    
    def get_interaction_history(self) -> List[UserInteraction]:
        """Retrieve the current interaction history."""
        return self.interaction_history.copy()
    
    def clear_history(self) -> None:
        """Clear all interaction history."""
        self.interaction_history = []
        self.exposure_risk_history = []
        self.entropy_history = []
        self.accuracy_history = []
        self.current_decay = self.config.decay_coefficient
    
    # =========================================================================
    # STEP 2: Adaptive Temporal Decay Function
    # =========================================================================
    
    def decay_kernel(self, time_delta: float, decay_coeff: Optional[float] = None) -> float:
        """
        Compute the exponential decay kernel.
        
        Equation (3): K(t_i, t) = exp(-lambda * (t - t_i))
        
        Args:
            time_delta: Time difference (current_time - interaction_time)
            decay_coeff: Decay coefficient (lambda). Uses current_decay if None.
            
        Returns:
            Decay weight in range (0, 1]
        """
        lam = decay_coeff if decay_coeff is not None else self.current_decay
        return np.exp(-lam * time_delta)
    
    def compute_decayed_preferences(self, current_time: float, 
                                   decay_coeff: Optional[float] = None) -> Dict[int, float]:
        """
        Compute decayed preference weights for all interacted items.
        
        Equation (4): p_decayed(t) = sum_i [ w_i * K(t_i, t) * item_i ]
        
        Args:
            current_time: Current timestamp
            decay_coeff: Optional override for decay coefficient
            
        Returns:
            Dictionary mapping item_id -> decayed cumulative weight
        """
        decayed_prefs: Dict[int, float] = {}
        
        for interaction in self.interaction_history:
            time_delta = current_time - interaction.timestamp
            if time_delta < 0:
                continue  # Future interactions are ignored
                
            decay_weight = self.decay_kernel(time_delta, decay_coeff)
            weighted_interaction = interaction.weight * decay_weight
            
            if interaction.item_id in decayed_prefs:
                decayed_prefs[interaction.item_id] += weighted_interaction
            else:
                decayed_prefs[interaction.item_id] = weighted_interaction
        
        return decayed_prefs
    
    # =========================================================================
    # STEP 3: Behavioral Entropy Measurement
    # =========================================================================
    
    def compute_normalized_weights(self, decayed_prefs: Dict[int, float]) -> Dict[int, float]:
        """
        Compute normalized preference weights (probability distribution).
        
        Equation (5): w_hat_i = p_decayed_i / sum_j(p_decayed_j)
        
        Args:
            decayed_prefs: Dictionary of decayed preference weights
            
        Returns:
            Normalized weights that sum to 1
        """
        total = sum(decayed_prefs.values())
        if total == 0:
            # Return uniform distribution if no preferences
            n = len(decayed_prefs)
            return {k: 1.0/n for k in decayed_prefs} if n > 0 else {}
        
        return {item_id: weight / total for item_id, weight in decayed_prefs.items()}
    
    def compute_entropy(self, normalized_weights: Dict[int, float]) -> float:
        """
        Compute Shannon entropy over the normalized interaction distribution.
        
        Equation (6): H = -sum_i [ w_hat_i * log(w_hat_i) ]
        
        Low entropy -> Over-personalization (concentrated on few items)
        High entropy -> Diverse exposure (uniform distribution)
        
        Returns entropy normalized to [0, 1] range.
        """
        if not normalized_weights:
            return 1.0  # Max entropy for empty history
        
        entropy = 0.0
        n = len(normalized_weights)
        
        for weight in normalized_weights.values():
            if weight > 1e-10:  # Avoid log(0)
                # Use log base 2, normalized by log(n) for [0,1] range
                entropy -= weight * np.log2(weight)
        
        # Normalize by maximum possible entropy (log2(n))
        max_entropy = np.log2(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return np.clip(normalized_entropy, 0.0, 1.0)
    
    # =========================================================================
    # STEP 4 & 5: Exposure Risk and Entropy-Regularized Ranking
    # =========================================================================
    
    def compute_exposure_risk(self, decayed_prefs: Dict[int, float]) -> float:
        """
        Compute the long-term exposure risk from personalization dominance.
        
        Equation (9): R = ||p_decayed||_2 / (sqrt(n) + epsilon)
        
        High norm -> High behavioral dominance -> High profiling risk
        
        Args:
            decayed_prefs: Dictionary of decayed preference weights
            
        Returns:
            Exposure risk score in [0, 1] range
        """
        if not decayed_prefs:
            return 0.0
        
        values = np.array(list(decayed_prefs.values()))
        l2_norm = np.linalg.norm(values, 2)
        
        # Normalize by sqrt(n) for scale-independence, add epsilon for stability
        n = len(values)
        epsilon = 1e-8
        normalized_risk = l2_norm / (np.sqrt(n) + epsilon)
        
        # Clip to [0, 1]
        return np.clip(normalized_risk, 0.0, 1.0)
    
    def compute_similarity(self, user_prefs: np.ndarray, item_features: np.ndarray) -> float:
        """
        Compute cosine similarity between user preference vector and item features.
        """
        if np.linalg.norm(user_prefs) == 0 or np.linalg.norm(item_features) == 0:
            return 0.0
        
        return np.dot(user_prefs, item_features) / (
            np.linalg.norm(user_prefs) * np.linalg.norm(item_features)
        )
    
    # =========================================================================
    # STEP 6: Adaptive Decay Update Rule
    # =========================================================================
    
    def update_decay_coefficient(self, exposure_risk: float, entropy: float) -> float:
        """
        Dynamically update the decay coefficient based on current state.
        
        Equation (10): lambda_{t+1} = lambda_t + alpha * (R_t - H_t)
        
        Logic:
        - If exposure_risk > entropy: Increase decay (forget more) because profiling is high
        - If entropy > exposure_risk: Decrease decay (remember more) because diversity is high
        
        Args:
            exposure_risk: Current exposure risk score
            entropy: Current behavioral entropy
            
        Returns:
            Updated decay coefficient (clipped to [min_decay, max_decay])
        """
        config = self.config
        
        # Update rule: increase decay if risk exceeds entropy target
        delta = exposure_risk - (1.0 - entropy)  # Risk vs. concentration penalty
        
        new_decay = self.current_decay + config.learning_rate * delta
        
        # Clip to valid range
        self.current_decay = np.clip(new_decay, config.min_decay, config.max_decay)
        
        return self.current_decay
    
    # =========================================================================
    # STEP 7: Final Ranking Score
    # =========================================================================
    
    def rank_items(self, candidate_items: List[ContentItem], 
                   current_time: float,
                   return_scores: bool = False) -> List[Tuple[ContentItem, float]]:
        """
        Rank candidate content items using the TEMD algorithm.
        
        Equation (11): S(item) = similarity(user, item) + popularity 
                                  - gamma * exposure_risk - beta * (1 - entropy)
        
        The ranking balances:
        1. Personal relevance (similarity with decayed preferences)
        2. Global popularity (cold-start handling)
        3. Privacy penalty (exposure risk term)
        4. Autonomy bonus (entropy regularization)
        
        Args:
            candidate_items: List of content items to rank
            current_time: Current timestamp
            return_scores: If True, return all scores for analysis
            
        Returns:
            List of (item, score) tuples sorted by score descending
        """
        config = self.config
        
        # Cold start: not enough interactions
        if len(self.interaction_history) < config.cold_start_threshold:
            # Sort by popularity only
            ranked = sorted(candidate_items, key=lambda x: x.popularity, reverse=True)
            result = []
            for item in ranked:
                details = {
                    'relevance': 0.0,
                    'popularity': item.popularity,
                    'privacy_penalty': 0.0,
                    'autonomy_bonus': 0.0,
                    'entropy': 1.0,
                    'exposure_risk': 0.0,
                    'decay_coefficient': self.current_decay
                }
                result.append((item, item.popularity, details))
            return result
        
        # Step 1: Compute decayed preferences
        decayed_prefs = self.compute_decayed_preferences(current_time)
        
        # Step 2: Compute normalized weights and entropy
        normalized_weights = self.compute_normalized_weights(decayed_prefs)
        entropy = self.compute_entropy(normalized_weights)
        
        # Step 3: Compute exposure risk
        exposure_risk = self.compute_exposure_risk(decayed_prefs)
        
        # Step 4: Build user preference vector from decayed preferences
        # (In practice, this would be the latent user representation)
        unique_items = list(decayed_prefs.keys())
        user_pref_vector = np.array([decayed_prefs.get(i, 0.0) for i in unique_items])
        
        # Normalize user preference vector
        if np.linalg.norm(user_pref_vector) > 0:
            user_pref_vector = user_pref_vector / np.linalg.norm(user_pref_vector)
        
        # Step 5: Compute scores for each candidate item
        scores = []
        
        for item in candidate_items:
            # Personal relevance: cosine similarity
            # For demo, we use a simplified feature-based similarity
            if len(item.features) > 0 and len(user_pref_vector) > 0:
                # Pad or truncate features to match dimension
                min_dim = min(len(item.features), len(user_pref_vector))
                relevance = np.dot(
                    item.features[:min_dim], 
                    user_pref_vector[:min_dim]
                )
            else:
                relevance = 0.0
            
            # Global popularity component (cold-start handling)
            popularity_score = item.popularity
            
            # Privacy penalty (exposure risk term)
            privacy_penalty = config.exposure_risk_weight * exposure_risk
            
            # Autonomy bonus (entropy regularization)
            # High entropy = diverse = good, so we add it
            autonomy_bonus = config.entropy_regularizer * entropy
            
            # Final score (Equation 11)
            score = relevance + popularity_score - privacy_penalty + autonomy_bonus
            
            scores.append((item, score, {
                'relevance': relevance,
                'popularity': popularity_score,
                'privacy_penalty': privacy_penalty,
                'autonomy_bonus': autonomy_bonus,
                'entropy': entropy,
                'exposure_risk': exposure_risk,
                'decay_coefficient': self.current_decay
            }))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Store metrics for history
        self.entropy_history.append(entropy)
        self.exposure_risk_history.append(exposure_risk)
        
        # Update decay coefficient adaptively
        self.update_decay_coefficient(exposure_risk, entropy)
        
        if return_scores:
            return [(item, score, details) for item, score, details in scores]
        
        return [(item, score) for item, score, _ in scores]
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_metrics(self) -> Dict:
        """Return current algorithm metrics."""
        return {
            'current_decay_coefficient': self.current_decay,
            'interaction_count': len(self.interaction_history),
            'entropy_history': self.entropy_history.copy(),
            'exposure_risk_history': self.exposure_risk_history.copy(),
            'accuracy_history': self.accuracy_history.copy()
        }
    
    def save_state(self, filepath: str) -> None:
        """Save algorithm state to JSON file."""
        state = {
            'config': self.config.to_dict(),
            'current_decay': self.current_decay,
            'interaction_history': [
                {
                    'item_id': i.item_id,
                    'timestamp': i.timestamp,
                    'weight': i.weight
                }
                for i in self.interaction_history
            ],
            'metrics': {
                'entropy_history': self.entropy_history,
                'exposure_risk_history': self.exposure_risk_history,
                'accuracy_history': self.accuracy_history
            }
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load algorithm state from JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = TEMDConfig(**state['config'])
        self.current_decay = state['current_decay']
        self.interaction_history = [
            UserInteraction(i['item_id'], i['timestamp'], i['weight'])
            for i in state['interaction_history']
        ]
        self.entropy_history = state['metrics']['entropy_history']
        self.exposure_risk_history = state['metrics']['exposure_risk_history']
        self.accuracy_history = state['metrics']['accuracy_history']


def create_sample_data(n_users: int = 100, 
                       n_items: int = 50,
                       n_interactions: int = 500,
                       feature_dim: int = 10,
                       random_seed: int = 42) -> Tuple[List, List, List]:
    """
    Generate synthetic social media data for testing.
    
    Returns:
        (users, items, interactions) where users is a list of TEMD instances,
        items is a list of ContentItem objects, and interactions is a list of
        (user_idx, interaction) tuples.
    """
    np.random.seed(random_seed)
    
    # Create content items with random features and popularity
    items = []
    for i in range(n_items):
        features = np.random.randn(feature_dim)
        features = features / np.linalg.norm(features)  # Normalize
        popularity = np.random.beta(2, 5)  # Most items have low popularity
        
        # Assign categories for diversity
        categories = ['tech', 'sports', 'news', 'entertainment', 'education']
        category = categories[i % len(categories)]
        
        items.append(ContentItem(
            item_id=i,
            features=features,
            popularity=popularity,
            category=category
        ))
    
    # Create users with TEMD instances
    users = []
    for u in range(n_users):
        # Vary user sensitivity (decay config)
        config = TEMDConfig(
            decay_coefficient=np.random.uniform(0.05, 0.3),
            entropy_regularizer=np.random.uniform(0.3, 0.7),
            exposure_risk_weight=np.random.uniform(0.2, 0.5),
            learning_rate=np.random.uniform(0.005, 0.02)
        )
        users.append(TemporalEthicalMemoryDecay(config))
    
    # Generate random interactions
    interactions = []
    current_time = 0.0
    
    for _ in range(n_interactions):
        user_idx = np.random.randint(0, n_users)
        item_idx = np.random.randint(0, n_items)
        
        # Time advances with each interaction
        current_time += np.random.exponential(0.5)
        
        # Weight based on interaction type
        weight = np.random.choice([1.0, 2.0, 3.0], p=[0.6, 0.3, 0.1])
        
        interaction = UserInteraction(
            item_id=item_idx,
            timestamp=current_time,
            weight=weight
        )
        
        users[user_idx].add_interaction(interaction)
        interactions.append((user_idx, interaction))
    
    return users, items, interactions


if __name__ == '__main__':
    # Demo: Basic algorithm test
    print("=" * 60)
    print("TEMD Algorithm - Basic Demo")
    print("=" * 60)
    
    # Create sample data
    users, items, interactions = create_sample_data(
        n_users=10, n_items=20, n_interactions=100, feature_dim=5
    )
    
    print(f"\nGenerated {len(users)} users, {len(items)} items, {len(interactions)} interactions")
    
    # Test ranking for first user
    user = users[0]
    current_time = max(i.timestamp for _, i in interactions) + 1.0
    
    print(f"\nUser 0 has {len(user.interaction_history)} interactions")
    print(f"Current decay coefficient: {user.current_decay:.4f}")
    
    # Rank items
    ranked = user.rank_items(items, current_time, return_scores=True)
    
    print(f"\nTop 5 Recommendations:")
    print("-" * 80)
    for i, (item, score, details) in enumerate(ranked[:5]):
        print(f"{i+1}. Item {item.item_id} ({item.category})")
        print(f"   Score: {score:.4f} | "
              f"Relevance: {details['relevance']:.4f} | "
              f"Popularity: {details['popularity']:.4f}")
        print(f"   Privacy Penalty: {details['privacy_penalty']:.4f} | "
              f"Autonomy Bonus: {details['autonomy_bonus']:.4f}")
    
    print(f"\nEntropy: {details['entropy']:.4f}")
    print(f"Exposure Risk: {details['exposure_risk']:.4f}")
    print(f"Updated Decay: {user.current_decay:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
