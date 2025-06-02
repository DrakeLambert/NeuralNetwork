import 'dart:math';
import 'package:neuralNet/matrix.dart';

// Export types and functions needed by training.dart
export 'matrix.dart' show Matrix;

/// Type definitions
typedef double ActivationFunction(double x);
typedef Layer = ({Matrix weights, Matrix biases});
typedef Network = List<Layer>;
typedef ActivatedLayer = ({
  Matrix inputs,
  Matrix weights,
  Matrix biases,
  Matrix weightedSums,
  Matrix activations
});
typedef ActivatedNetwork = List<ActivatedLayer>;

/// ReLU (Rectified Linear Unit) activation function
double reLU(double x) => max(0, x);

/// Derivative of the ReLU activation function
double reLUDerivative(double x) => x > 0 ? 1 : 0;

/// Hyperbolic tangent (tanh) activation function
double tanh(double x) => (exp(x) - exp(-x)) / (exp(x) + exp(-x));

/// Derivative of the tanh activation function
double tanhDerivative(double x) {
  final tanhX = tanh(x);
  return 1 - (tanhX * tanhX);
}

/// Performs forward propagation through the network, returning the state
/// of each layer including inputs, weights, biases, and activations.
ActivatedNetwork forwardPropagation(
    Matrix inputs, Network network, ActivationFunction activationFunction) {
  if (network case []) {
    return [];
  }
  var [layer, ...restLayers] = network;
  final weightedSums = layer.weights * inputs + layer.biases;
  final activations = weightedSums.map(activationFunction);

  return [
    (
      inputs: inputs,
      weights: layer.weights,
      biases: layer.biases,
      weightedSums: weightedSums,
      activations: activations
    ),
    ...forwardPropagation(activations, restLayers, activationFunction)
  ];
}

/// Creates a new neural network with random weights and biases.
/// The network will have the specified number of inputs and the given layer sizes.
Network createNetwork(int inputCount, List<int> layerSizes, {int? seed}) {
  final random = Random(seed);
  final nextRandom = () => random.nextDouble() * 2 - 1;

  Network createNetworkRecursive(int inputCount, List<int> layerSizes) =>
      switch (layerSizes) {
        [] => [],
        [var layerSize, ...var restLayerSizes] => [
            (
              biases: Matrix.fromColumn(
                  List.generate(layerSize, (_) => nextRandom())),
              weights: Matrix.generate(layerSize, inputCount,
                  (rowIndex, columnIndex) => nextRandom())
            ),
            ...createNetworkRecursive(layerSize, restLayerSizes)
          ]
      };

  return createNetworkRecursive(inputCount, layerSizes);
}

/// Performs backpropagation on the network using the computed gradients
/// to update weights and biases.
Network backPropagation(
    ActivatedNetwork activatedNetwork,
    Matrix desiredActivations,
    ActivationFunction activationFunctionDerivative,
    double stepSize,
    {bool showCalculations = false}) {
  Network backPropagation(ActivatedNetwork activatedNetwork, Matrix dCdA) {
    switch (activatedNetwork) {
      case [...var restLayers, var layer]:
        final dAdZ = layer.weightedSums.map(activationFunctionDerivative);
        final dCdB = dCdA.hadamardProduct(dAdZ);
        final dCdW = Matrix.fromColumns(layer.inputs.columns.first
            .map((input) => (dCdB * input).columns.first)
            .toList());

        if (showCalculations) {
          print("\n==== LAYER ${restLayers.length} ====");
          print("\ndCdA:\n$dCdA");
          print("\ndAdZ:\n$dAdZ");
          print("\ndCdB:\n$dCdB");
          print("\ndCdW:\n$dCdW");
        }

        final dCdA_next = layer.weights.transpose() * dCdB;
        return [
          ...backPropagation(restLayers, dCdA_next),
          (
            biases: layer.biases + (dCdB * -stepSize),
            weights: layer.weights + (dCdW * -stepSize)
          )
        ];
      default:
        return [];
    }
  }

  final lastLayer = activatedNetwork.last;
  final dCdA = (lastLayer.activations - desiredActivations) * 2;
  return backPropagation(activatedNetwork, dCdA);
}

/// Calculates the mean squared error between expected and actual outputs
double meanSquaredError(Matrix expected, Matrix actual) {
  final squaredDifferences =
      (expected - actual).map((e) => pow(e, 2).toDouble());
  final sumOfSquares = squaredDifferences.columns
      .map((column) => column.reduce((sum, next) => sum + next))
      .reduce((sum, next) => sum + next);

  return sumOfSquares / (expected.rowCount * expected.columnCount);
}

/// Utility function to zip two iterables together
Iterable<(TA, TB)> zip<TA, TB>(Iterable<TA> a, Iterable<TB> b) sync* {
  final aIterator = a.iterator;
  final bIterator = b.iterator;
  while (aIterator.moveNext() && bIterator.moveNext()) {
    yield (aIterator.current, bIterator.current);
  }
}

/// Prints a human-readable representation of the network's weights and biases
void printNetwork(Network network) {
  for (var layer = 0; layer < network.length; layer++) {
    print("\n==== LAYER $layer ====");
    print("\nWeights:\n${network[layer].weights}");
    print("\nBiases:\n${network[layer].biases}");
  }
}
