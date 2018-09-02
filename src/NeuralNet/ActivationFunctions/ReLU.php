<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;

/**
 * ReLU
 *
 * Rectified Linear Units output only the positive part of its inputs and are
 * analogous to a half-wave rectifiers in electrical engineering.
 *
 * References:
 * [1] V. Nair et al. (2011). Rectified Linear Units Improve Restricted
 * Boltzmann Machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ReLU implements Rectifier
{
    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0., INF];
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
            return $value > 0. ? $value : 0.;
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
        return $computed->map(function ($activation) {
            return $activation > 0. ? 1. : 0.;
        });
    }
}
