<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts;

use NDArray;

/**
 * Derivative based on input / output buffer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface IOBufferDerivative
{
    /**
     * Calculate the derivative of the activation.
     *
     * @param NDArray $input Input matrix
     * @param NDArray $output Output matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray;
}
