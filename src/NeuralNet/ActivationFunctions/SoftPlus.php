<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

/**
 * Soft Plus
 *
 * A smooth approximation of the ReLU function whose output is constrained to be
 * positive.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SoftPlus implements Rectifier
{
    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0, INF];
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
            return log(1.0 + exp($value));
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
            return 1.0 / (1.0 + exp(-$activation));
        });
    }
}
