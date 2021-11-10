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
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function activate(Matrix $input) : Matrix;

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $output
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $input, Matrix $output) : Matrix;
}
