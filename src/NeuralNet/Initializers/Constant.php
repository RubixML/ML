<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Constant
 *
 * Initialize the parameter to a user specified constant value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Initializer
{
    /**
     * The value to initialize the parameter to.
     *
     * @var float
     */
    protected float $value;

    /**
     * @param float $value
     * @throws InvalidArgumentException
     */
    public function __construct(float $value = 0.0)
    {
        if (is_nan($value)) {
            throw new InvalidArgumentException('Cannot initialize'
                . ' weight values to NaN.');
        }

        $this->value = $value;
    }

    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @internal
     *
     * @param int<0,max> $fanIn
     * @param int<0,max> $fanOut
     * @return Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix
    {
        return Matrix::fill($this->value, $fanOut, $fanIn);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Constant (value: {$this->value})";
    }
}
