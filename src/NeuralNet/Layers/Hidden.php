<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

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
     * Calculate the gradient and update the parameters of the layer.
     *
     * @internal
     *
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return \Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred;
}
