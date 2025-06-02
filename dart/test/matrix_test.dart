import 'package:neuralNet/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('matrix', () {
    test('fromRows', () {
      final a = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];

      final actual = Matrix.fromRows(a);

      expect(actual.columns[0][0], 1.0);
      expect(actual.columns[0][1], 4.0);
      expect(actual.columns[1][0], 2.0);
      expect(actual.columns[1][1], 5.0);
      expect(actual.columns[2][0], 3.0);
      expect(actual.columns[2][1], 6.0);

      expect(actual.rows[0][0], 1.0);
      expect(actual.rows[0][1], 2.0);
      expect(actual.rows[0][2], 3.0);
      expect(actual.rows[1][0], 4.0);
      expect(actual.rows[1][1], 5.0);
      expect(actual.rows[1][2], 6.0);

      expect(actual.row(0), a[0]);
      expect(actual.row(1), a[1]);

      expect(actual.column(0), [1.0, 4.0]);
      expect(actual.column(1), [2.0, 5.0]);
      expect(actual.column(2), [3.0, 6.0]);
    });

    test('dot product', () {
      final a = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];
      final b = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
      ];
      final expected = [
        [22.0, 28.0],
        [49.0, 64.0]
      ];

      final actual = Matrix.fromRows(a) * Matrix.fromRows(b);

      expect(actual.columns, Matrix.fromRows(expected).columns);
    });

    test('multiply matrix', () {
      final a = [
        [1.0, 2.0],
        [4.0, 5.0]
      ];
      final b = [
        [1.0],
        [3.0]
      ];
      final expected = [
        [1 * 1 + 2 * 3.0],
        [4 * 1 + 5 * 3.0]
      ];

      final actual = Matrix.fromRows(a) * Matrix.fromRows(b);

      expect(actual.columns, Matrix.fromRows(expected).columns);
    });

    test('add matrix', () {
      final a = [
        [1.0, 2.0],
        [4.0, 5.0]
      ];
      final b = [
        [6.0, 7.0],
        [8.0, 9.0]
      ];
      final expected = [
        [7.0, 9.0],
        [12.0, 14.0]
      ];

      final actual = Matrix.fromRows(a) + Matrix.fromRows(b);

      expect(actual.columns, Matrix.fromRows(expected).columns);
    });

    test('hadamard product', () {
      final a = [
        [1.0, 2.0],
        [4.0, 5.0]
      ];
      final b = [
        [6.0, 7.0],
        [8.0, 9.0]
      ];
      final expected = [
        [6.0, 14.0],
        [32.0, 45.0]
      ];

      final actual = Matrix.fromRows(a).hadamardProduct(Matrix.fromRows(b));

      expect(actual.rows, expected);
    });
  });
}
