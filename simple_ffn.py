import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFN(nn.Module):
    """A basic feed-forward network with one hidden layer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer returns logits; softmax is applied by the loss or at inference
        return self.fc2(x)


def make_dummy_data(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate simple clustered data for a toy classification task."""
    torch.manual_seed(seed)
    # Create class centers and add noise around them
    centers = torch.randn(num_classes, input_dim) * 2.0
    labels = torch.randint(0, num_classes, (num_samples,))
    features = centers[labels] + 0.5 * torch.randn(num_samples, input_dim)
    return features, labels


def train(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> None:
    """Train the model with SGD and cross-entropy loss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    epochs = 100

    model.train()
    for epoch in range(1, epochs + 1):
        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)

        # Backward pass and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")


def evaluate(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> None:
    """Evaluate accuracy on a dataset."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main() -> None:
    # Hyperparameters for the toy example
    input_dim = 4
    hidden_dim = 8
    output_dim = 3

    # Generate training and evaluation data
    train_x, train_y = make_dummy_data(num_samples=200, input_dim=input_dim, num_classes=output_dim)
    test_x, test_y = make_dummy_data(num_samples=80, input_dim=input_dim, num_classes=output_dim, seed=123)

    # Initialize and train the model
    model = SimpleFFN(input_dim, hidden_dim, output_dim)
    train(model, train_x, train_y)

    # Evaluate on new data
    evaluate(model, test_x, test_y)


if __name__ == "__main__":
    main()
