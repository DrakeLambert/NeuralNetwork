# Neural Network Implementation

A neural network implementation with in Dart and F# (unfinished).

## Features

- Feed-forward neural networks with configurable layers
- Multiple activation functions (ReLU, tanh)
- Matrix-based operations for efficient computation
- Backpropagation training with configurable parameters
- Batch training support
- Configurable learning rates and convergence criteria

## Example

The Dart implementation includes an XOR problem demonstration showing how to:

- Create a network with multiple hidden layers
- Train using gradient descent
- Test predictions on new inputs

## Getting Started

### Dart Implementation

```dart
// Create a network (2 inputs, 4 hidden, 2 hidden, 1 output)
final network = createNetwork(2, [4, 2, 1]);

// Train the network
final result = trainUntilConvergence(
  network,
  trainingBatch,
  tanh,
  tanhDerivative,
  stepSize: 0.05,
  maxTrainingPasses: 10000
);
```

### F# Implementation

```fsharp
// Create a network (3 inputs, [38, 20, 10] hidden layers)
let net = Net.create 3 [38; 20; 10]

// Evaluate inputs through the network
[100.0..100.0..300.0]
|> eval sigmoid net
```
