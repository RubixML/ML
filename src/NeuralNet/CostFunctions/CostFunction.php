<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;

interface CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array;

    /**
     * Compute the cost.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix;

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @param  \Rubix\ML\Other\Structures\Matrix  $delta
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix;
}
