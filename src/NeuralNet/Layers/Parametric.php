<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Generator;

/**
 * Parametric
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
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
     * Return the accumulated gradients of the parameters of the layer.
     *
     * @return Generator<array{Parameter, Tensor<int|float|array>}>
     */
    public function gradients() : Generator;

    /**
     * Reset the accumulated gradients of the layer.
     */
    public function resetGradients() : void;

    /**
     * Restore the parameters on the layer from an associative array.
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void;
}
