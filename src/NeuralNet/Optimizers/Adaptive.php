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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Adaptive extends Optimizer
{
    /**
     * Set the data type of the parameter cache.
     *
     * @param string $datatype
     */
    public function setCacheDataType(string $datatype) : void;

    /**
     * Warm the parameter cache.
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void;
}
