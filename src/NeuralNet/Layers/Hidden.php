<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Deferred;

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
     * Calculate the gradient for the previous layer and record the gradients of the parameters of this layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @return Deferred
     */
    public function back(Deferred $prevGradient) : Deferred;
}
