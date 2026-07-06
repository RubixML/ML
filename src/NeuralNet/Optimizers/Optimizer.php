<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use NDArray;
use Rubix\ML\NeuralNet\Parameter;
use Stringable;

/**
 * Optimizer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Optimizer extends Stringable
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray;
}
