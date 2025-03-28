<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

use const Rubix\ML\EPSILON;

/**
 * Softmax
 *
 * The Softmax function is a generalization of the Sigmoid function that squashes
 * each activation between 0 and 1, and all activations add up to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Softmax extends Sigmoid
{
    /**
     * Compute the activation.
     *
     * @internal
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function activate(Matrix $input) : Matrix
    {
        return NumPower::exp($input - NumPower::max($input)) / NumPower::sum(NumPower::exp($input));
    }

    public function differentiate(Matrix $input, Matrix $output) : Matrix
    {
        $s = 1 / (1 + NumPower::exp(-$input));
        
        return NumPower::diag($s) - NumPower::outer($s, $s);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Softmax';
    }
}
