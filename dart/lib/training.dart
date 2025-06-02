import 'package:neuralNet/matrix.dart';
import 'package:neuralNet/network.dart';

/// Trains a network using a batch of training samples.
/// Returns a new network with updated weights and biases.
Network train(
    Network network,
    Iterable<({Matrix inputs, Matrix expected})> trainingBatch,
    ActivationFunction activationFunction,
    ActivationFunction activationFunctionDerivative,
    double stepSize) {
  return trainingBatch.fold(network, (updatedNetwork, trainingSample) {
    final activatedNetwork = forwardPropagation(
        trainingSample.inputs, updatedNetwork, activationFunction);
    return backPropagation(activatedNetwork, trainingSample.expected,
        activationFunctionDerivative, stepSize);
  });
}

/// Result of training a network until convergence
typedef TrainingResult = ({Network network, double error, int passCount});

/// Trains a network until either the error is below the tolerance
/// or the maximum number of passes is reached.
TrainingResult trainUntilConvergence(
  Network initialNetwork,
  Iterable<({Matrix inputs, Matrix expected})> trainingBatch,
  ActivationFunction activationFunction,
  ActivationFunction activationFunctionDerivative, {
  double stepSize = 0.05,
  int maxTrainingPasses = 1000,
  double errorTolerance = 0.0,
  void Function(int pass, double error)? onPassComplete,
}) {
  var currentNetwork = initialNetwork;
  var pass = 0;

  while (pass < maxTrainingPasses) {
    final error =
        calculateBatchError(currentNetwork, trainingBatch, activationFunction);
    onPassComplete?.call(pass, error);

    if (error <= errorTolerance) {
      return (network: currentNetwork, error: error, passCount: pass);
    }

    currentNetwork = train(currentNetwork, trainingBatch, activationFunction,
        activationFunctionDerivative, stepSize);

    pass++;
  }

  final finalError =
      calculateBatchError(currentNetwork, trainingBatch, activationFunction);

  return (network: currentNetwork, error: finalError, passCount: pass);
}

/// Calculates the mean squared error across a batch of training samples
double calculateBatchError(
    Network network,
    Iterable<({Matrix inputs, Matrix expected})> batch,
    ActivationFunction activationFunction) {
  final batchResults = batch.map((trainingSample) {
    final actual =
        forwardPropagation(trainingSample.inputs, network, activationFunction)
            .last
            .activations;
    return (expected: trainingSample.expected, actual: actual);
  }).reduce((rest, next) => (
        expected: Matrix.fromColumns(
            [...rest.expected.columns, ...next.expected.columns]),
        actual:
            Matrix.fromColumns([...rest.actual.columns, ...next.actual.columns])
      ));

  return meanSquaredError(batchResults.expected, batchResults.actual);
}

/// Takes a list of networks and averages their weights and biases.
/// Useful for combining results from parallel training.
Network average(List<Network> networks) {
  final sum = networks.reduce((sum, next) => zip(sum, next)
      .map((layer) => (
            weights: layer.$1.weights + layer.$2.weights,
            biases: layer.$1.biases + layer.$2.biases
          ))
      .toList());

  return sum
      .map((layer) => (
            weights: layer.weights / networks.length,
            biases: layer.biases / networks.length
          ))
      .toList();
}
