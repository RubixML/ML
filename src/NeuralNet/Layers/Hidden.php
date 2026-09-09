<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Tensor\Tensor;

/**
 * Hidden
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Hidden extends Layer
{
    /**
     * Calculate the gradient for the previous layer and return the gradients of the parameters
     * of this layer along with a deferred computation of the gradient of the previous layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @return array{Deferred, list<array{Parameter, Tensor<int|float|array>}>}
     */
    public function back(Deferred $prevGradient) : array;
}
