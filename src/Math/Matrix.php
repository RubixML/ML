<?php

namespace Rubix\Engine\Math;

use InvalidArgumentException;
use OutOfBoundsException;
use IteratorAggregate;
use ArrayIterator;
use ArrayAccess;

class Matrix implements ArrayAccess, IteratorAggregate
{
    /**
     * The scalar values stored in the matrix.
     *
     * @var array
     */
    protected $values;

    /**
     * Build a vector from an array of values. If row is set to true the output
     * will be a row vector otherwise it will be a column vector.
     *
     * @param  array  $values
     * @param  bool  $row
     * @return self
     */
    public static function vector(array $values = [], bool $row = false) : self
    {
        if ($row === true) {
            return new static([$values]);
        } else {
            return new static(array_map(function ($value) {
                return [$value];
            }, $values));
        }
    }

    /**
     * @param  array  $values
     * @return void
     */
    public function __construct(array $values = [[]])
    {
        if (!is_array($values[0])) {
            $values = [$values];
        }

        foreach (range(0, count($values) - 1) as $i) {
            if (count($values[$i]) !== count($values[0])) {
                throw new InvalidArgumentException('The length of every column must be equal.');
            }
        }

        $this->values = $values;
    }

    /**
     * The dimensions of the matrix.
     *
     * @return array
     */
    public function dimensions() : array
    {
        return [$this->rows(), $this->columns()];
    }

    /**
     * The number of rows in the matrix.
     *
     * @return int
     */
    public function rows() : int
    {
        return count($this->values);
    }

    /**
     * The number of columns in the matrix.
     *
     * @return int
     */
    public function columns() : int
    {
        return count($this->values[0]);
    }

    /**
     * @return array
     */
    public function values() : array
    {
        return $this->values;
    }

    /**
     * Select a row and return a vector.
     *
     * @param  int  $n
     * @return array
     */
    public function row(int $n) : self
    {
        if ($n < 0 || $n > $this->rows()) {
            throw new OutOfBoundsException('Row number is out of range.');
        }

        return new static($this->values[$n]);
    }

    /**
     * Select a column and return a vector.
     *
     * @param  int  $n
     * @return array
     */
    public function column(int $n) : self
    {
        if ($n < 0 || $n > $this->columns()) {
            throw new OutOfBoundsException('Column number is out of range.');
        }

        return new static(array_column($this->values, $n));
    }

    /**
     * Add this matrix to another matrix. O(M*N)
     *
     * @param  \Rubix\Engine\Math\Matrix  $matrix
     * @return self
     */
    public function add(self $matrix) : self
    {
        $sum = [[]];

        foreach (range(0, $this->rows() - 1) as $i) {
            foreach (range(0, $this->columns() - 1) as $j) {
                $sum[$i][$j] = $this->values[$i][$j] + $matrix[$i][$j];
            }
        }

        return new static($sum);
    }

    /**
     * Subtract this matrix from another matrix. O(M*N)
     *
     * @param  \Rubix\Engine\Math\Matrix  $matrix
     * @return self
     */
    public function subtract(self $matrix) : self
    {
        $difference = [[]];

        foreach (range(0, $this->rows() - 1) as $i) {
            foreach (range(0, $this->columns() - 1) as $j) {
                $difference[$i][$j] = $this->values[$i][$j] - $matrix[$i][$j];
            }
        }

        return new static($difference);
    }

    /**
     * Multiply this matrix with another matrix. O(N^3)
     *
     * @param  \Rubix\Engine\Math\Matrix  $matrix
     * @return self
     */
    public function multiply(self $matrix) : self
    {
        if ($this->columns() !== $matrix->rows()) {
            throw new InvalidArgumentException('Matrices cannot be multiplied because they are inconsistent.');
        }

        $product = [[]];

        foreach ($this->values as $row => $rowValues) {
            foreach (range(0, $matrix->columns() - 1) as $column) {
                $columnValues = $matrix->column($column)->values()[0];

                $sigma = 0;

                foreach ($rowValues as $i => $value) {
                    $sigma += $value * $columnValues[$i];
                }

                $product[$row][$column] = $sigma;
            }
        }

        return new static($product);
    }

    /**
     * Multiply the matrix by a scalar value. O(M*N)
     *
     * @param  mixed  $scalar
     * @return self
     */
    public function multiplyByScalar($scalar) : self
    {
        if (!is_numeric($scalar)) {
            throw new InvalidArgumentException('Scalar value must be numeric type, ' . gettype($scalar) . ' found.');
        }

        $product = [[]];

        foreach (range(0, $this->rows() - 1) as $i) {
            foreach (range(0, $this->columns() - 1) as $j) {
                $product[$i][$j] = $this->values[$i][$j] * $scalar;
            }
        }

        return new static($product);
    }

    /**
     * Divide the matrix by a scalar value. O(M*N)
     *
     * @param  mixed  $scalar
     * @return self
     */
    public function divideByScalar($scalar) : self
    {
        if (!is_numeric($scalar)) {
            throw new InvalidArgumentException('Scalar value must be numeric type, ' . gettype($scalar) . ' found.');
        }

        $quotient = [[]];

        foreach (range(0, $this->rows() - 1) as $i) {
            foreach (range(0, $this->columns() - 1) as $j) {
                $quotient[$i][$j] = $this->values[$i][$j] / $scalar;
            }
        }

        return new static($quotient);
    }

    /**
     * Calculate the dot product of two column vectors and return the cooresponding
     * matrix.
     *
     * @param  \Rubix\Engine\Math\Matrix  $vector
     * @return self
     */
    public function dot(self $vector) : self
    {
        if ($this->columns() !== 1 || $vector->columns() !== 1) {
            throw new InvalidArgumentException('Both operands must be column vectors.');
        }

        return $this->multiply($vector->transpose());
    }

    /**
     * Rotate the matrix on its diagonal. ex. turn a column vector into a row
     * vector.
     *
     * @return self
     */
    public function transpose() : self
    {
        if ($this->rows() === 1) {
            $matrix = array_map(function ($column) {
                return [$column];
            }, $this->values[0]);
        } else {
            $matrix = array_map(null, ...$this->values);
        }

        return new static($matrix);
    }

    /**
     * Returns an identity matrix with the same dimensions as this one.
     *
     * @return self
     */
    public function identity() : self
    {
        $matrix = array_fill(0, $this->rows(), array_fill(0, $this->columns(), 0));

        foreach (range(0, $this->rows() - 1) as $i) {
            $matrix[$i][$i] = 1;
        }

        return new self($matrix);
    }

    /**
     * Is this a square matrix? i.e. # of rows equals # of columns.
     *
     * @return bool
     */
    public function isSquare() : bool
    {
        return $this->rows() === $this->columns();
    }

    /**
     * @param  mixed  $offset
     * @param  mixed  $value
     * @return void
     */
    public function offsetSet($offset, $value) : void
    {
        throw new RuntimeException('Cannot modify matrix value, matrices are immutable.');
    }

    /**
     * @param  mixed  $offset
     * @return bool
     */
    public function offsetExists($offset) : bool
    {
        return isset($this->values[$offset]);
    }

    /**
     * @param  mixed  $offset
     * @return void
     */
    public function offsetUnset($offset) : void
    {
        throw new RuntimeException('Cannot modify matrix value, matrices are immutable.');
    }

    /**
     * @param  mixed  $offset
     * @return mixed
     */
    public function offsetGet($offset)
    {
        if ($this->offsetExists($offset)) {
            return $this->values[$offset];
        }
    }

    /**
     * Get an iterator for the rows in the matrix.
     *
     * @return \ArrayIterator
     */
    public function getIterator()
    {
        return new ArrayIterator($this->values);
    }
}
