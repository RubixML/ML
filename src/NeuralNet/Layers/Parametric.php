<?php

namespace Rubix\ML\NeuralNet\Layers;

use Generator;
use Rubix\ML\NeuralNet\Parameter;

/**
 * Parametric
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Parametric
{
    /**
     * Return the parameters of the layer.
     *
     * @return Generator<Parameter>
     */
    public function parameters() : Generator;

    /**
     * Restore the parameters on the layer from an associative array.
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void;
}
