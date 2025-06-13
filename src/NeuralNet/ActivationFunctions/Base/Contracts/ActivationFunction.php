<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts;

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
     * @param NDArray $input Input matrix
     * @return NDArray Output matrix
     */
    public function activate(NDArray $input) : NDArray;

    /**
     * Calculate the derivative of the activation.
     *
     * @param NDArray $input Input matrix
     * @param NDArray $output Output matrix
     * @return NDArray Derivative matrix
     */
    //public function differentiate(NDArray $input, NDArray $output) : NDArray;
}
