<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NDArray;
use Stringable;

/**
 * Activation Function
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
interface ActivationFunction extends Stringable
{
    /**
     * Compute the activation.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray;

    /**
     * Calculate the derivative of the activation for backpropagation.
     *
     * @param NDArray $input
     * @param NDArray $output
     * @return NDArray
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray;
}
