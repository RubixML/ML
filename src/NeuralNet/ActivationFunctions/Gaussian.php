<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;

/**
 * Gaussian
 *
 * The Gaussian activation function is a type of Radial Basis Function (*RBF*)
 * whose activation depends only on the distance the input is from the origin.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Gaussian implements ActivationFunction
{
    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0., 1.];
    }

    /**
     * Compute the output value.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map(function ($value) {
            return exp(-($value ** 2));
        });
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @param  \Rubix\ML\Other\Structures\Matrix  $computed
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $z->multiply($computed)->multiplyScalar(-2.);
    }
}
