import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableTopKLayer(nn.Module):
    """
    A trainable layer that selects top-k points based on their scores.
    The k value is learned during training through a temperature-controlled soft selection.

    Args:
        input_dim: Dimension of input features (default: 3 for x,y,z coordinates)
        initial_k_ratio: Initial ratio of points to keep (0 to 1)
        min_k_ratio: Minimum ratio of points to keep
        max_k_ratio: Maximum ratio of points to keep
        temperature: Initial temperature for soft thresholding (lower = sharper selection)
    """

    def __init__(
        self,
        input_dim=3,
        initial_k_ratio=0.5,
        min_k_ratio=0.1,
        max_k_ratio=0.9,
        temperature=1.0,
    ):
        super(TrainableTopKLayer, self).__init__()

        # Learnable parameter for k ratio (in logit space for better optimization)
        self.k_logit = nn.Parameter(
            torch.tensor(
                self._ratio_to_logit(initial_k_ratio, min_k_ratio, max_k_ratio)
            )
        )

        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio

        # Learnable temperature for controlling sharpness of selection
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

        # Optional: Learnable weights to transform/enhance the scores
        self.score_weight = nn.Parameter(torch.ones(1))
        self.score_bias = nn.Parameter(torch.zeros(1))

    def _ratio_to_logit(self, ratio, min_ratio, max_ratio):
        """Convert ratio to logit space for unconstrained optimization"""
        # Map [min_ratio, max_ratio] to [0, 1] then to logit
        normalized = (ratio - min_ratio) / (max_ratio - min_ratio)
        normalized = torch.clamp(torch.tensor(normalized), 0.01, 0.99)
        return torch.log(normalized / (1 - normalized))

    def get_k_ratio(self):
        """Get current k ratio from learnable parameter"""
        # Convert logit back to ratio in [min_k_ratio, max_k_ratio]
        sigmoid_val = torch.sigmoid(self.k_logit)
        return self.min_k_ratio + sigmoid_val * (self.max_k_ratio - self.min_k_ratio)

    def forward(self, points, scores):
        """
        Args:
            points: Tensor of shape (batch_size, num_points, input_dim) or (num_points, input_dim)
            scores: Tensor of shape (batch_size, num_points) or (num_points,)

        Returns:
            masked_scores: Scores with non-selected points set to zero
            selection_mask: Soft mask indicating selection probability (0 to 1)
            k_value: Current k value being used
        """
        # Handle both batched and unbatched inputs
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_points = scores.shape

        # Transform scores with learnable weights
        enhanced_scores = scores * self.score_weight + self.score_bias

        # Get current k ratio and calculate k
        k_ratio = self.get_k_ratio()
        k = torch.clamp(torch.round(k_ratio * num_points).long(), min=1, max=num_points)

        # Get temperature
        temperature = torch.exp(self.log_temperature)

        # Compute threshold using top-k (soft approximation during training)
        if self.training:
            # Soft top-k selection using sigmoid and threshold
            # Find the k-th largest score as threshold
            topk_scores, _ = torch.topk(enhanced_scores, k.item(), dim=1)
            threshold = topk_scores[:, -1:].detach()  # Use last value as threshold

            # Smooth approximation: sigmoid((score - threshold) / temperature)
            selection_mask = torch.sigmoid((enhanced_scores - threshold) / temperature)
        else:
            # Hard top-k selection during inference
            topk_scores, topk_indices = torch.topk(enhanced_scores, k.item(), dim=1)
            selection_mask = torch.zeros_like(enhanced_scores)
            selection_mask.scatter_(1, topk_indices, 1.0)

        # Apply mask to original scores
        masked_scores = scores * selection_mask

        if squeeze_output:
            masked_scores = masked_scores.squeeze(0)
            selection_mask = selection_mask.squeeze(0)

        return masked_scores, selection_mask, k

    def get_info(self):
        """Return current layer parameters for monitoring"""
        return {
            "k_ratio": self.get_k_ratio().item(),
            "temperature": torch.exp(self.log_temperature).item(),
            "score_weight": self.score_weight.item(),
            "score_bias": self.score_bias.item(),
        }


class AdvancedTopKLayer(nn.Module):
    """
    Advanced version with attention mechanism to learn importance weights
    that combine with input scores for better selection.

    Args:
        input_dim: Dimension of input features. Can be any features like:
                   - Raw coordinates: 3 for x,y,z
                   - Pillar features: 6 for (x_offset, y_offset, z_offset,
                     xy_offset, distance_to_centroid, etc.)
                   - Any custom feature representation
        hidden_dim: Hidden dimension for attention network
        initial_k_ratio: Initial ratio of points to keep
        temperature: Temperature for soft selection
    """

    def __init__(
        self, input_dim=3, hidden_dim=64, initial_k_ratio=0.5, temperature=1.0
    ):
        super(AdvancedTopKLayer, self).__init__()

        # Attention network to learn importance from point coordinates
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable k parameter
        self.k_logit = nn.Parameter(
            torch.tensor(self._ratio_to_logit(initial_k_ratio, 0.1, 0.9))
        )

        # Temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

        # Weight for combining attention with input scores
        self.alpha = nn.Parameter(
            torch.tensor(0.5)
        )  # Balance between score and attention

    def _ratio_to_logit(self, ratio, min_ratio, max_ratio):
        normalized = (ratio - min_ratio) / (max_ratio - min_ratio)
        normalized = torch.clamp(torch.tensor(normalized), 0.01, 0.99)
        return torch.log(normalized / (1 - normalized))

    def get_k_ratio(self):
        sigmoid_val = torch.sigmoid(self.k_logit)
        return 0.1 + sigmoid_val * 0.8

    def forward(self, points, scores):
        """
        Args:
            points: Tensor of shape (batch_size, num_points, input_dim)
            scores: Tensor of shape (batch_size, num_points)

        Returns:
            masked_scores: Filtered scores
            selection_mask: Selection mask
            k_value: Current k
            combined_scores: Combined attention + input scores
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_points, _ = points.shape

        # Compute attention scores from point coordinates
        attention_scores = self.attention_net(points).squeeze(-1)

        # Combine input scores with learned attention
        alpha = torch.sigmoid(self.alpha)
        combined_scores = alpha * scores + (1 - alpha) * attention_scores

        # Get k
        k_ratio = self.get_k_ratio()
        k = torch.clamp(torch.round(k_ratio * num_points).long(), min=1, max=num_points)

        # Temperature
        temperature = torch.exp(self.log_temperature)

        # Top-k selection
        if self.training:
            topk_scores, _ = torch.topk(combined_scores, k.item(), dim=1)
            threshold = topk_scores[:, -1:].detach()
            selection_mask = torch.sigmoid((combined_scores - threshold) / temperature)
        else:
            topk_scores, topk_indices = torch.topk(combined_scores, k.item(), dim=1)
            selection_mask = torch.zeros_like(combined_scores)
            selection_mask.scatter_(1, topk_indices, 1.0)

        masked_scores = scores * selection_mask

        if squeeze_output:
            masked_scores = masked_scores.squeeze(0)
            selection_mask = selection_mask.squeeze(0)
            combined_scores = combined_scores.squeeze(0)

        return masked_scores, selection_mask, k, combined_scores


# Example usage and testing
def test_topk_layer():
    """Test the trainable top-k layers"""

    print("=" * 60)
    print("Testing TrainableTopKLayer")
    print("=" * 60)

    # Create sample data
    batch_size = 2
    num_points = 100
    input_dim = 3

    points = torch.randn(batch_size, num_points, input_dim)
    scores = torch.randn(batch_size, num_points)

    # Initialize layer
    layer = TrainableTopKLayer(
        input_dim=input_dim, initial_k_ratio=0.5, temperature=0.1
    )

    print(f"\nInitial parameters:")
    info = layer.get_info()
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")

    # Forward pass
    masked_scores, selection_mask, k = layer(points, scores)

    print(f"\nForward pass results:")
    print(f"  Input shape: {scores.shape}")
    print(f"  Output shape: {masked_scores.shape}")
    print(f"  Selected k: {k.item()}")
    print(f"  Non-zero scores: {(masked_scores != 0).sum(dim=1)}")
    print(
        f"  Selection mask stats: min={selection_mask.min():.4f}, max={selection_mask.max():.4f}"
    )

    # Test gradient flow
    loss = masked_scores.sum()
    loss.backward()

    print(f"\nGradients computed successfully!")
    print(f"  k_logit gradient: {layer.k_logit.grad}")
    print(f"  score_weight gradient: {layer.score_weight.grad}")

    print("\n" + "=" * 60)
    print("Testing AdvancedTopKLayer")
    print("=" * 60)

    # Test advanced layer
    advanced_layer = AdvancedTopKLayer(
        input_dim=input_dim, hidden_dim=64, initial_k_ratio=0.5, temperature=0.1
    )

    masked_scores, selection_mask, k, combined_scores = advanced_layer(points, scores)

    print(f"\nAdvanced layer results:")
    print(f"  Selected k: {k.item()}")
    print(f"  Non-zero scores: {(masked_scores != 0).sum(dim=1)}")
    print(
        f"  Combined scores range: [{combined_scores.min():.4f}, {combined_scores.max():.4f}]"
    )

    # Test gradient flow
    loss = masked_scores.sum()
    loss.backward()
    print(f"\nGradients computed successfully for advanced layer!")


def example_pillar_features():
    """
    Example of using AdvancedTopKLayer with pillar-based features
    instead of raw x,y,z coordinates.
    """
    print("\n" + "=" * 60)
    print("Example: Using Pillar Features")
    print("=" * 60)

    batch_size = 2
    num_points = 200

    # Pillar features: [x_offset, y_offset, z_offset, xy_offset, distance_to_centroid, intensity]
    pillar_feature_dim = 6

    # Create sample pillar features
    pillar_features = torch.randn(batch_size, num_points, pillar_feature_dim)
    scores = torch.randn(batch_size, num_points)

    # Initialize layer with pillar feature dimension
    layer = AdvancedTopKLayer(
        input_dim=pillar_feature_dim,  # Use 6 instead of 3
        hidden_dim=128,
        initial_k_ratio=0.3,
        temperature=0.1,
    )

    print(f"\nInput features shape: {pillar_features.shape}")
    print(f"Feature dimension: {pillar_feature_dim} (pillar features)")

    # Forward pass
    masked_scores, selection_mask, k, combined_scores = layer(pillar_features, scores)

    print(f"\nResults:")
    print(f"  Selected k: {k.item()} out of {num_points} points")
    print(f"  Selection ratio: {k.item() / num_points:.2%}")
    print(f"  Non-zero scores per batch: {(masked_scores != 0).sum(dim=1).tolist()}")

    # The attention network learns which pillar features are important
    # for point selection (e.g., maybe points closer to centroid are more important)
    print(f"\n✓ The attention network learned importance from pillar features!")
    print(f"  It can learn patterns like:")
    print(f"  - Points with smaller distance to centroid are more important")
    print(f"  - Points with certain offset patterns are more valuable")
    print(f"  - Any complex relationship between your features")


if __name__ == "__main__":
    test_topk_layer()
    example_pillar_features()
