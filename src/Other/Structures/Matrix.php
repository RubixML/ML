<?php

namespace Rubix\ML\Other\Structures;

use InvalidArgumentException;
use IteratorAggregate;
use RuntimeException;
use ArrayIterator;
use ArrayAccess;
use Countable;

/**
 * Matrix
 *
 * Two dimensional tensor with integer and/or floating point elements.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Matrix implements ArrayAccess, IteratorAggregate, Countable
{
    /**
     * The 2 dimensional array that holds the values of the matrix.
     *
     * @var array
     */
    protected $a = [
        //
    ];

    /**
     * The number of rows in the matrix.
     *
     * @var int
     */
    protected $m;

    /**
     * The number of columns in the matrix.
     *
     * @var int
     */
    protected $n;

    /**
     * Factory method to build a new matrix from an array.
     *
     * @param  array  $a
     * @param  bool  $validate
     * @return self
     */
    public static function build(array $a, bool $validate = true) : self
    {
        return new self($a, $validate);
    }

    /**
     * Return an identity matrix with the given dimensions.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function identity(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Dimensionality must be'
                . ' greater than 0 along both axis.');
        }

        $a = [[]];

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $a[$i][$j] = $i === $j ? 1 : 0;
            }
        }

        return new self($a, false);
    }

    /**
     * Return a zero matrix with the given dimensions.
     *
     * @param  int  $m
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function zeros(int $m, int $n) : self
    {
        if ($m < 1 or $n < 1) {
            throw new InvalidArgumentException('Dimensionality must be'
                . ' greater than 0 along both axis.');
        }

        return new self(array_fill(0, $m, array_fill(0, $n, 0)), false);
    }

    /**
     * Return a one matrix with the given dimensions.
     *
     * @param  int  $m
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function ones(int $m, int $n) : self
    {
        if ($m < 1 or $n < 1) {
            throw new InvalidArgumentException('Dimensionality must be'
                . ' greater than 0 along both axis.');
        }

        return new self(array_fill(0, $m, array_fill(0, $n, 1)), false);
    }

    /**
     * Build a diagonal matrix with the value of each element along the
     * diagonal and 0s everywhere else.
     *
     * @param  array  $elements
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function diagonal(array $elements) : self
    {
        $n = count($elements);

        if ($n === 0) {
            throw new InvalidArgumentException('Dimensionality must be'
                . ' greater than 0 along both axis.');
        }

        $a = [[]];

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $a[$i][$j] = ($i === $j) ? $elements[$i] : 0;
            }
        }

        return new self($a, false);
    }

    /**
     * @param  array[]  $a
     * @param  bool  $validate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $a, bool $validate = true)
    {
        if ($validate === true) {
            $a = array_values($a);

            $n = is_array($a[0]) ? count($a[0]) : 0;

            foreach ($a as &$row) {
                if (count($row) !== $n) {
                    throw new InvalidArgumentException('The number of columns'
                        . ' must be equal for all rows.');
                }

                foreach ($row as $value) {
                    if (!is_int($value) and !is_float($value)) {
                        throw new InvalidArgumentException('Matrix element must'
                            . ' be an integer or float, '
                            . gettype($value) . ' found.');
                    }
                }

                $row = array_values($row);
            }
        }

        $this->a = $a;
        $this->m = count($a);
        $this->n = isset($a[0]) ? count($a[0]) : 0;
    }

    /**
     * Return a tuple with the dimensionality of the matrix.
     *
     * @return int[]
     */
    public function shape() : array
    {
        return [$this->m, $this->n];
    }

    /**
     * Return the number of elements in the matrix.
     *
     * @return int
     */
    public function size() : int
    {
        return $this->m * $this->n;
    }

    /**
     * Return the number of rows in the matrix.
     *
     * @return int
     */
    public function m() : int
    {
        return $this->m;
    }

    /**
     * Return the number of columns in the matrix.
     *
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }

    /**
     * Return a row form the matrix.
     *
     * @param  int  $index
     * @return array
     */
    public function row(int $index) : array
    {
        return $this->offsetGet($index);
    }

    /**
     * Return a column from the matrix.
     *
     * @param  int  $index
     * @return array
     */
    public function column(int $index) : array
    {
        return array_column($this->a, $index);
    }

    /**
     * Transpose the matrix.
     *
     * @return self
     */
    public function transpose() : self
    {
        if ($this->m > 1) {
            $b = array_map(null, ...$this->a);
        } else {
            $b = [];

            for ($i = 0; $i < $this->n; $i++) {
                $b[$i] = array_column($this->a, $i);
            }
        }

        return new self($b, false);
    }

    /**
     * Run a function over all of the elements in the matrix.
     *
     * @param  callable  $fn
     * @return self
     */
    public function map(callable $fn) : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $fn($value);
            }
        }

        return new self($b, true);
    }

    /**
     * Take the dot product of this matrix and another matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function dot(Matrix $b) : self
    {
        if ($b->m() !== $this->n) {
            throw new InvalidArgumentException('Matrix dimensions do not'
                . ' match.');
        }

        $bT = $b->transpose();
        $c = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($bT as $j => $column) {
                $sigma = 0;

                foreach ($row as $k => $value) {
                    $sigma += $value * $column[$k];
                }

                $c[$i][$j] = $sigma;
            }
        }

        return new self($c, false);
    }

    /**
     * Return the elementwise product between this matrix and another matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function multiply(Matrix $b) : self
    {
        if ($b->m() !== $this->m) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of rows.');
        }

        if ($b->n() !== $this->n) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of columns.');
        }

        $c = [[]];

        foreach ($this->a as $i => $rowA) {
            $rowB = $b[$i];

            foreach ($rowA as $j => $value) {
                $c[$i][$j] = $value * $rowB[$j];
            }
        }

        return new self($c, false);
    }

    /**
     * Return the division of two elements, elementwise.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function divide(Matrix $b) : self
    {
        if ($b->m() !== $this->m) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of rows.');
        }

        if ($b->n() !== $this->n) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of columns.');
        }

        $c = [[]];

        foreach ($this->a as $i => $rowA) {
            $rowB = $b[$i];

            foreach ($rowA as $j => $value) {
                $c[$i][$j] = $value / $rowB[$j];
            }
        }

        return new self($c, false);
    }

    /**
     * Add this matrix together with another matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function add(Matrix $b) : self
    {
        if ($b->m() !== $this->m) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of rows.');
        }

        if ($b->n() !== $this->n) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of columns.');
        }

        $c = [[]];

        foreach ($this->a as $i => $rowA) {
            $rowB = $b[$i];

            foreach ($rowA as $j => $value) {
                $c[$i][$j] = $value + $rowB[$j];
            }
        }

        return new self($c, false);
    }

    /**
     * Subtract this matrix from another matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function subtract(Matrix $b) : self
    {
        if ($b->m() !== $this->m) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of rows.');
        }

        if ($b->n() !== $this->n) {
            throw new InvalidArgumentException('Matrices have different number'
                . ' of columns.');
        }

        $c = [[]];

        foreach ($this->a as $i => $rowA) {
            $rowB = $b[$i];

            foreach ($rowA as $j => $value) {
                $c[$i][$j] = $value - $rowB[$j];
            }
        }

        return new self($c, false);
    }

    /**
     * Return the elementwise reciprocal of the matrix.
     *
     * @return self
     */
    public function reciprocal() : self
    {
        return self::ones(...$this->shape())->divide($this);
    }

    /**
     * Raise this matrix to the power of the elementwise entry in another
     * matrix.
     *
     * @param \Rubix\ML\Other\Structures\Matrix  $b
     * @return self
     */
    public function power(Matrix $b) : self
    {
        $c = [[]];

        foreach ($this->a as $i => $rowA) {
            $rowB = $b[$i];

            foreach ($rowA as $j => $value) {
                $c[$i][$j] = $value ** $rowB[$j];
            }
        }

        return new self($c, false);
    }

    /**
     * Multiply this matrix by a scalar.
     *
     * @param  mixed  $scalar
     * @throws \InvalidArgumentException
     * @return self
     */
    public function scalarMultiply($scalar) : self
    {
        if (!is_int($scalar) and !is_float($scalar)) {
            throw new InvalidArgumentException('Factor must be an integer or'
                . ' float ' . gettype($scalar) . ' found.');
        }

        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $value * $scalar;
            }
        }

        return new self($b, false);
    }

    /**
     * Divide this matrix by a scalar.
     *
     * @param  mixed  $scalar
     * @throws \InvalidArgumentException
     * @return self
     */
    public function scalarDivide($scalar) : self
    {
        if (!is_int($scalar) and !is_float($scalar)) {
            throw new InvalidArgumentException('Factor must be an integer or'
                . ' float ' . gettype($scalar) . ' found.');
        }

        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $value / $scalar;
            }
        }

        return new self($b, false);
    }

    /**
     * Add this matrix by a scalar.
     *
     * @param  mixed  $scalar
     * @throws \InvalidArgumentException
     * @return self
     */
    public function scalarAdd($scalar) : self
    {
        if (!is_int($scalar) and !is_float($scalar)) {
            throw new InvalidArgumentException('Factor must be an integer or'
                . ' float ' . gettype($scalar) . ' found.');
        }

        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $value + $scalar;
            }
        }

        return new self($b, false);
    }

    /**
     * Subtract this matrix by a scalar.
     *
     * @param  mixed  $scalar
     * @throws \InvalidArgumentException
     * @return self
     */
    public function scalarSubtract($scalar) : self
    {
        if (!is_int($scalar) and !is_float($scalar)) {
            throw new InvalidArgumentException('Factor must be an integer or'
                . ' float ' . gettype($scalar) . ' found.');
        }

        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $value - $scalar;
            }
        }

        return new self($b, false);
    }

    /**
     * Sum the matrix along a specified axis.
     *
     * @param  int  $axis
     * @throws \InvalidArgumentException
     * @return self
     */
    public function sum(int $axis = 0)
    {
        if ($axis < 0 or $axis > 1) {
            throw new InvalidArgumentException('Axis can be 0 or 1'
                . (string) $axis . ' given.');
        }
        $b = [[]];

        if ($axis === 0) {
            foreach ($this->transpose() as $i => $column) {
                $b[0][$i] = array_sum($column);
            }
        } else {
            foreach ($this->a as $i => $row) {
                $b[$i][0] = array_sum($row);
            }
        }

        return new self($b, false);
    }

    /**
     * The sum of all the elements in a row of the matrix.
     *
     * @param  int  $index
     * @return float
     */
    public function rowSum(int $index) : float
    {
        return array_sum($this->offsetGet($index));
    }

    /**
     * The sum of all the elements in a column of the matrix.
     *
     * @param  int  $index
     * @return float
     */
    public function columnSum(int $index) : float
    {
        return array_sum($this->column($index));
    }

    /**
     * Return the absolute value of each element in the matrix.
     *
     * @return self
     */
    public function abs() : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = abs($value);
            }
        }

        return new self($b, false);
    }

    /**
     * Return the square of the matrix elementwise.
     *
     * @return self
     */
    public function square() : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $value ** 2;
            }
        }

        return new self($b, false);
    }

    /**
     * Return the square root of the matrix.
     *
     * @return self
     */
    public function sqrt() : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = $value ** 0.5;
            }
        }

        return new self($b, false);
    }

    /**
     * Return the exponential of the matrix.
     *
     * @return self
     */
    public function exp() : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = M_E ** $value;
            }
        }

        return new self($b, false);
    }

    /**
     * Return the logarithm of the matrix in specified base.
     *
     * @param  float  $base
     * @return self
     */
    public function log(float $base = M_E) : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = log($value, $base);
            }
        }

        return new self($b, false);
    }

    /**
     * Round the elements in the matrix to a given decimal place.
     *
     * @param  int  $precision
     * @return self
     */
    public function round(int $precision = 0) : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = round($value, $precision);
            }
        }

        return new self($b, false);
    }

    /**
     * Round the elements in the matrix down to the nearest integer.
     *
     * @return self
     */
    public function floor() : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = floor($value);
            }
        }

        return new self($b, false);
    }

    /**
     * Round the elements in the matrix up to the nearest integer.
     *
     * @return self
     */
    public function ceil() : self
    {
        $b = [[]];

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $b[$i][$j] = ceil($value);
            }
        }

        return new self($b, false);
    }

    /**
     * Return the L1 norm of the matrix.
     *
     * @return float
     */
    public function l1Norm() : float
    {
        $norm = 0.;

        foreach ($this->transpose() as $column) {
            $norm = max($norm, array_sum(array_map('abs', $column)));
        }

        return $norm;
    }

    /**
     * Return the L2 norm of the matrix.
     *
     * @return float
     */
    public function l2Norm() : float
    {
        $norm = 0.;

        foreach ($this->a as $row) {
            foreach ($row as $value) {
                $norm += $value ** 2;
            }
        }

        return sqrt($norm);
    }

    /**
     * Retrn the infinity norm of the matrix.
     *
     * @return float
     */
    public function infinityNorm() : float
    {
        $norm = 0.;

        foreach ($this->a as $row) {
            $norm = max($norm, array_sum(array_map('abs', $row)));
        }

        return $norm;
    }

    /**
     * Return the max norm of the matrix.
     *
     * @return float
     */
    public function maxNorm() : float
    {
        $norm = 0.;

        foreach ($this->a as $i => $row) {
            foreach ($row as $j => $value) {
                $norm = max($norm, abs($value));
            }
        }

        return $norm;
    }

    /**
     * Exclude a row from the matrix.
     *
     * @param  int  $index
     * @return self
     */
    public function rowExclude(int $index) : self
    {
        $b = $this->a;

        unset($b[$index]);

        return new self($b, false);
    }

    /**
     * Exclude a column from the matrix.
     *
     * @param  int  $index
     * @return self
     */
    public function columnExclude(int $index) : self
    {
        $b = $this->a;

        foreach ($b as $i => &$row) {
            unset($row[$index]);

            $row = array_values($row);
        }

        return new self($b, false);
    }

    /**
     * Attach matrix b above this matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function augmentAbove(Matrix $b) : self
    {
        if ($b->n() !== $this->n) {
            throw new InvalidArgumentException('Matrices must have the same'
                . ' number of columns.');
        }

        return new self(array_merge($b->asArray(), $this->a), false);
    }

    /**
     * Attach matrix b below this matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function augmentBelow(Matrix $b) : self
    {
        if ($b->n() !== $this->n) {
            throw new InvalidArgumentException('Matrices must have the same'
                . ' number of columns.');
        }

        return new self(array_merge($this->a, $b->asArray()), false);
    }

    /**
     * Attach matrix b to the left of this matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function augmentLeft(Matrix $b) : self
    {
        if ($b->m() !== $this->m()) {
            throw new InvalidArgumentException('Matrices must have the same'
                . ' number of rows.');
        }

        $c = [];

        foreach ($this->a as $i => $row) {
            $c[] = array_merge($b[$i], $row);
        }

        return new self($c);
    }

    /**
     * Attach matrix b to the left of this matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $b
     * @throws \InvalidArgumentException
     * @return self
     */
    public function augmentRight(Matrix $b) : self
    {
        if ($b->m() !== $this->m()) {
            throw new InvalidArgumentException('Matrices must have the same'
                . ' number of rows.');
        }

        $c = [];

        foreach ($this->a as $i => $row) {
            $c[] = array_merge($row, $b[$i]);
        }

        return new self($c);
    }

    /**
     * Repeat the matrix m times along the vertival axis and n times along the
     * horizontal axis.
     *
     * @param  int  $m
     * @param  int  $n
     * @return self
     */
    public function repeat(int $m = 1, int $n = 1) : self
    {
        if ($m < 1 or $n < 1) {
            throw new InvalidArgumentException('Cannot repeat less than 1 row'
                . ' or column.');
        }

        $m -= 1;
        $n -= 1;

        $a = $this;
        $b = $a;

        for ($i = 0; $i < $n; $i++) {
            $a = $a->augmentRight($b);
        }

        $c = $a;

        for ($i = 0; $i < $m; $i++) {
            $a = $a->augmentBelow($c);
        }

        return $a;
    }

    /**
     * Return the elements of the matrix in a 2-d array.
     *
     * @return array
     */
    public function asArray() : array
    {
        return $this->a;
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->size();
    }

    /**
     * @param  mixed  $index
     * @param  array  $values
     * @throws \RuntimeException
     * @return void
     */
    public function offsetSet($index, $values) : void
    {
        throw new RuntimeException('Matrix cannot be mutated directly.');
    }

    /**
     * Does a given column exist in the matrix.
     *
     * @param  mixed  $index
     * @return bool
     */
    public function offsetExists($index) : bool
    {
        return isset($this->a[$index]);
    }

    /**
     * @param  mixed  $index
     * @throws \RuntimeException
     * @return void
     */
    public function offsetUnset($index) : void
    {
        throw new RuntimeException('Matrix cannot be mutated directly.');
    }

    /**
     * Return a row from the matrix at the given index.
     *
     * @param  mixed  $index
     * @throws \InvalidArgumentException
     * @return array
     */
    public function offsetGet($index) : array
    {
        if (!isset($this->a[$index])) {
            throw new InvalidArgumentException('Element not found at index'
                . (string) $index . '.');
        }

        return $this->a[$index];
    }

    /**
     * Get an iterator for the rows in the matrix.
     *
     * @return \ArrayIterator
     */
    public function getIterator()
    {
        return new ArrayIterator($this->a);
    }
}
