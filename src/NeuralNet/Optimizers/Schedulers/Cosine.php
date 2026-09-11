<?php

namespace Rubix\ML\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Cosine
 *
 * A cosine annealing learning rate schedule that smoothly decays the rate from
 * a starting rate down to an ending rate over a fixed number of steps, then
 * holds the end rate for the remainder of training.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cosine implements Scheduler
{
    /**
     * The starting rate.
     *
     * @var float
     */
    protected float $start;

    /**
     * The ending rate.
     *
     * @var float
     */
    protected float $end;

    /**
     * The number of steps taken to reach the ending rate.
     *
     * @var int
     */
    protected int $steps;

    /**
     * The precomputed learning rate.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The range between the starting and ending rates.
     *
     * @var float
     */
    protected float $range;

    /**
     * The number of steps taken so far.
     *
     * @var int
     */
    protected int $t = 0;

    /**
     * @param float $start
     * @param float $end
     * @param int $steps
     * @throws InvalidArgumentException
     */
    public function __construct(float $start = 0.01, float $end = 0.0001, int $steps = 1000)
    {
        if ($start <= 0.0) {
            throw new InvalidArgumentException('Starting rate must be'
                . " greater than 0, $start given.");
        }

        if ($end <= 0.0) {
            throw new InvalidArgumentException('The ending rate must be'
                . " greater than 0, $end given.");
        }

        if ($steps < 1) {
            throw new InvalidArgumentException('The number of steps must be'
                . " greater than 0, $steps given.");
        }

        $this->start = $start;
        $this->end = $end;
        $this->range = $start - $end;
        $this->steps = $steps;
        $this->rate = $start;
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

        $fraction = min(1.0, $this->t / $this->steps);

        $this->rate = $this->end + $this->range * (1 + cos(M_PI * $fraction)) / 2;
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
        return "Cosine (start: {$this->start}, end: {$this->end}, steps: {$this->steps})";
    }
}
