<?php

namespace Rubix\ML\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Constant
 *
 * A learning rate schedule that outputs a constant learning rate for the
 * entire duration of training.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Scheduler
{
    /**
     * The constant learning rate.
     *
     * @var float
     */
    protected float $rate;

    /**
     * @param float $rate
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        $this->rate = $rate;
    }

    /**
     * Return the current learning rate.
     *
     * @return float
     */
    public function rate() : float
    {
        return $this->rate;
    }

    /**
     * Advance the learning-rate schedule by one batch.
     *
     * @internal
     */
    public function tick() : void
    {
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Constant (rate: {$this->rate})";
    }
}
