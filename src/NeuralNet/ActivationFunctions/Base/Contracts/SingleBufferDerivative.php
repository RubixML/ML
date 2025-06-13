<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts;

use NDArray;

/**
 * Derivative based on input buffer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
interface SingleBufferDerivative extends Derivative
{
    /**
     * Calculate the derivative of the single parameter.
     *
     * @param NDArray $x Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $x): NDArray;
}