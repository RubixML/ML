<?php

namespace Rubix\ML\NeuralNet\Layers\Base\Contracts;

use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Layer;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;

/**
 * Hidden
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Hidden extends Layer
{
    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @param Optimizer $optimizer
     * @return Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred;
}
