<?php

namespace Rubix\ML\NeuralNet\Optimizers\Base;

use Rubix\ML\NeuralNet\Parameters\Parameter;

/**
 * Adaptive
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Adaptive extends Optimizer
{
    /**
     * Warm the parameter cache.
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void;
}
