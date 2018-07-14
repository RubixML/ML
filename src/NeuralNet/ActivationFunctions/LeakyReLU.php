<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

/**
 * Leaky ReLU
 *
 * Leaky Rectified Linear Units are functions that output x when x > 0 or a
 * small leakage value when x < 0. The amount of leakage is controlled by the
 * user-specified parameter.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeakyReLU implements Rectifier
{
    /**
     * The amount of leakage as a ratio of the input value to allow to pass
     * through when not activated.
     *
     * @var float
     */
    protected $leakage;

    /**
     * @param  float  $leakage
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $leakage = 0.01)
    {
        if ($leakage < 0 or $leakage > 1) {
            throw new InvalidArgumentException('Leakage parameter must be'
                . ' between 0 and 1.');
        }

        $this->leakage = $leakage;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [-INF, INF];
    }

    /**
     * Compute the output value.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map(function ($value) {
            return $value >= 0.0 ? $value : $this->leakage * $value;
        });
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @param  \MathPHP\LinearAlgebra\Matrix  $computed
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map(function ($activation) {
            return $activation >= 0.0 ? 1.0 : $this->leakage;
        });
    }
}
