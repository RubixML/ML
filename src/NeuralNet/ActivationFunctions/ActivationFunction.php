<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Stringable;

/**
 * Activation Function
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface ActivationFunction extends Stringable
{
    /**
     * Compute the activation.
     *
     * @internal
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix;

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix;
}
