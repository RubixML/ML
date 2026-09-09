<?php

namespace Rubix\ML\NeuralNet\Optimizers\Schedulers;

use Stringable;

/**
 * Scheduler
 *
 * A learning rate schedule paired with an optimizer. The scheduler produces
 * a learning rate that is advanced by one "batch" each time the network
 * completes a single forward/backward pass.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Scheduler extends Stringable
{
    /**
     * Return the current learning rate.
     *
     * @return float
     */
    public function rate() : float;

    /**
     * Advance the learning-rate schedule by one batch.
     *
     * @internal
     */
    public function tick() : void;
}
