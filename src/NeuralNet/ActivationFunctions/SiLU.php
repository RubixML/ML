<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * SiLU
 *
 * Sigmoid Linear Units are smooth and non-monotonic rectified activation functions. Their inputs are weighted by
 * the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism.
 *
 * References:
 * [1] S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
 * Reinforcement Learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SiLU implements ActivationFunction
{
    /**
     * Compute the activation.
     *
     * @internal
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix
    {
        return $x->divide($x->negate()->exp()->add(1.0));
    }

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix
    {
        $sigmoid = $x->negate()->exp()->add(1.0)->reciprocal();

        $xHat = $x->multiply($sigmoid)->multiply($sigmoid->negate()->add(1.0));

        return $sigmoid->add($xHat);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'SiLU';
    }
}
