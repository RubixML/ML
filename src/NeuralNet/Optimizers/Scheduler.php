<?php

namespace Rubix\ML\NeuralNet\Optimizers;

/**
 * Scheduler
 *
 * A learning rate scheduler that advances its internal schedule by one
 * "batch" each time the network completes a single forward/backward pass.
 * The schedule is global to the optimizer (not per-parameter) so it must be
 * ticked once per batch, independent of how many parameters are updated.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Scheduler extends Optimizer
{
    /**
     * Advance the learning-rate schedule by one batch.
     *
     * @internal
     */
    public function tick() : void;
}
