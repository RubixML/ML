<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;

/**
 * Adaptive
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
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
