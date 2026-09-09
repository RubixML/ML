<?php

namespace Rubix\ML\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Step Decay
 *
 * A step-wise learning rate schedule that reduces the learning rate by a factor
 * of the decay parameter whenever it reaches a new *floor*. The number of
 * steps needed to reach a new floor is defined by the *steps* parameter.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class StepDecay implements Scheduler
{
    /**
     * The initial learning rate.
     *
     * @var float
     */
    protected float $initialRate;

    /**
     * The size of every floor in steps.
     *
     * @var int
     */
    protected int $losses;

    /**
     * The factor to decrease the learning rate by over a period of k steps.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The precomputed learning rate.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The number of steps taken so far.
     *
     * @var int
     */
    protected int $t = 0;

    /**
     * @param float $initialRate
     * @param int $losses
     * @param float $decay
     * @throws InvalidArgumentException
     */
    public function __construct(float $initialRate = 0.01, int $losses = 100, float $decay = 1e-3)
    {
        if ($initialRate <= 0.0) {
            throw new InvalidArgumentException('Initial learning rate must be'
                . " greater than 0, $initialRate given.");
        }

        if ($losses < 1) {
            throw new InvalidArgumentException('The number of steps per'
                . " floor must be greater than 0, $losses given.");
        }

        if ($decay < 0.0) {
            throw new InvalidArgumentException('Decay rate must be'
                . " positive, $decay given.");
        }

        $this->initialRate = $initialRate;
        $this->losses = $losses;
        $this->decay = $decay;
        $this->rate = $initialRate;
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
        ++$this->t;

        $floor = floor($this->t / $this->losses);

        $this->rate = $this->initialRate * (1.0 / (1.0 + $floor * $this->decay));
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
        return "Step Decay (rate: {$this->initialRate}, steps: {$this->losses}, decay: {$this->decay})";
    }
}
