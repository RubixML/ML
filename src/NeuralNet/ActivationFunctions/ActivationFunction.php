<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;

interface ActivationFunction
{
    const EPSILON = 1e-10;

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array;

    /**
     * Compute the output value.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $z) : Matrix;

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @param  \Rubix\ML\Other\Structures\Matrix  $computed
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix;
}
