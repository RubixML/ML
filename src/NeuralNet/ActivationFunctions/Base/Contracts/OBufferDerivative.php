<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts;

use NDArray;

/**
 * Derivative based on output buffer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface OBufferDerivative
{
    /**
     * Calculate the derivative of the activation.
     *
     * @param NDArray $output Output matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $output) : NDArray;
}
