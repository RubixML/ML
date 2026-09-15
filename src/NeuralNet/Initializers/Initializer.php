<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Rubix\ML\NeuralNet\Parameter;
use Stringable;

/**
 * Initializer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Initializer extends Stringable
{
    /**
     * Initialize a parameter tensor given its target shape.
     *
     * A one-dimensional shape of [$n] initializes a column vector of length $n.
     * A two-dimensional shape of [$m, $n] initializes a matrix with shape ($m, $n)
     * where $m is the number of outputs and $n is the number of inputs.
     *
     * @internal
     *
     * @param int[] $shape
     * @return Parameter
     */
    public function initialize(array $shape) : Parameter;
}
