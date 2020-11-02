<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

/**
 * Initializer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @internal
     *
     * @param int $fanIn
     * @param int $fanOut
     * @return \Tensor\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
