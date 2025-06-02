class Matrix {
  final List<List<double>> _columns;
  final int rowCount;
  final int columnCount;

  Matrix.generate(this.rowCount, this.columnCount,
      double Function(int rowIndex, int columnIndex) generator)
      : _columns = List.generate(
            columnCount,
            (columnIndex) => List.generate(
                rowCount, (rowIndex) => generator(rowIndex, columnIndex)));

  Matrix.fromRows(List<List<double>> rows)
      : this.generate(rows.length, rows.length > 0 ? rows.first.length : 0,
            (rowIndex, columnIndex) => rows[rowIndex][columnIndex]);

  Matrix.fromColumns(List<List<double>> columns)
      : this.generate(
            columns.length > 0 ? columns.first.length : 0,
            columns.length,
            (rowIndex, columnIndex) => columns[columnIndex][rowIndex]);

  Matrix.fromColumn(List<double> values)
      : rowCount = values.length,
        columnCount = 1,
        _columns = [values];

  Matrix.fromRow(List<double> values)
      : rowCount = 1,
        columnCount = values.length,
        _columns = List.from(values.map((value) => [value]));

  Matrix.fromScalar(double value)
      : rowCount = 1,
        columnCount = 1,
        _columns = [
          [value]
        ];

  List<double> column(int columnIndex) => _columns[columnIndex];

  List<double> row(int rowIndex) =>
      List.from(_columns.map((column) => column[rowIndex]));

  List<List<double>> get rows => List.generate(rowCount, row);

  List<List<double>> get columns => _columns;

  Matrix map(double Function(double value) toValue) => Matrix.generate(
      rowCount,
      columnCount,
      (rowIndex, columnIndex) => toValue(_columns[columnIndex][rowIndex]));

  Matrix operator *(Object other) {
    if (other case num other) return map((value) => value * other);

    if (other case Matrix other) {
      if (columnCount != other.rowCount)
        throw ArgumentError(
            'this Matrix must have the same number of columns as other Matrix has rows');

      return Matrix.generate(rowCount, other.columnCount,
          (rowIndex, columnIndex) {
        final thisRow = row(rowIndex);
        final otherColumn = other.column(columnIndex);
        var sum = 0.0;
        for (var i = 0; i < thisRow.length; i++) {
          sum += thisRow[i] * otherColumn[i];
        }
        return sum;
      });
    }

    throw ArgumentError('other must be of type num or Matrix');
  }

  Matrix operator /(Object other) {
    if (other case num other) return map((value) => value / other);

    throw ArgumentError('other must be of type num');
  }

  Matrix _combine(
      Object other, double Function(double value, num other) combine) {
    if (other case num other) return map((value) => combine(value, other));

    if (other case Matrix other) {
      if (columnCount != other.columnCount || rowCount != other.rowCount)
        throw ArgumentError(
            'this Matrix must have the same shape as other Matrix');

      return Matrix.generate(
          rowCount,
          columnCount,
          (rowIndex, columnIndex) => combine(columns[columnIndex][rowIndex],
              other.columns[columnIndex][rowIndex]));
    }

    throw ArgumentError('other must be of type num or Matrix');
  }

  Matrix operator +(Object other) =>
      _combine(other, (value, other) => value + other);

  Matrix operator -(Object other) =>
      _combine(other, (value, other) => value - other);

  Matrix hadamardDivision(Matrix other) =>
      _combine(other, (value, other) => value / other);

  Matrix hadamardProduct(Matrix other) =>
      _combine(other, (value, other) => value * other);

  Matrix transpose() => Matrix.fromRows(columns);

  String toString() => rows
      .map((row) => row.map((x) => x.toStringAsFixed(3).padLeft(7)).join(' '))
      .reduce((lines, line) => "$lines\n$line");
}
