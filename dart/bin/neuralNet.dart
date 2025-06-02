import 'package:neuralNet/network.dart';
import 'package:neuralNet/training.dart';

void main() {
  // Define XOR training data
  final trainingBatch = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
  ].map((trainingSample) => (
        inputs: Matrix.fromColumn(trainingSample.$1),
        expected: Matrix.fromScalar(trainingSample.$2),
      ));

  // Create and train network
  final network = createNetwork(2, [4, 2, 1],
      seed: 1); // 2 inputs, 4 hidden, 2 hidden, 1 output
  final result = trainUntilConvergence(
    network,
    trainingBatch,
    tanh,
    tanhDerivative,
    stepSize: 0.05,
    maxTrainingPasses: 10000,
    errorTolerance: 0.0,
    onPassComplete: (pass, error) {
      print("Pass: $pass Error: $error");
    },
  );

  print(
      "\nTraining completed in ${result.passCount} passes with final error ${result.error}\n");
  print("Final Network:\n");
  printNetwork(result.network);

  // Test the network on all XOR inputs
  print("\nTesting XOR predictions:\n");
  final testInputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
  ];

  for (final input in testInputs) {
    final prediction =
        forwardPropagation(Matrix.fromColumn(input), result.network, tanh)
            .last
            .activations
            .columns
            .first
            .first;
    print(
        "Input: (${input[0]}, ${input[1]}) → Output: ${prediction.toStringAsFixed(3)}");
  }
}
