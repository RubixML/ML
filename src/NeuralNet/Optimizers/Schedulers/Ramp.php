<?php

namespace Rubix\ML\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function min;

/**
 * Ramp
 *
 * A linear learning rate schedule that ramps the rate from a start rate to an
 * end rate over a fixed number of steps, then holds the end rate for the
 * remainder of training. It can be used to warm up the rate from a low start
 * to a high target or to cool it down over time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ramp implements Scheduler
{
    /**
     * The starting learning rate.
     *
     * @var float
     */
    protected float $start;

    /**
     * The ending learning rate.
     *
     * @var float
     */
    protected float $end;

    /**
     * The number of steps taken to reach the ending learning rate.
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
    public function __construct(float $start = 0.001, float $end = 0.01, int $steps = 1000)
    {
        if ($start <= 0.0) {
            throw new InvalidArgumentException('Starting learning rate must be'
                . " greater than 0, $start given.");
        }

        if ($end <= 0.0) {
            throw new InvalidArgumentException('Ending learning rate must be'
                . " greater than 0, $end given.");
        }

        if ($steps < 1) {
            throw new InvalidArgumentException('The number of steps must be'
                . " greater than 0, $steps given.");
        }

        $this->start = $start;
        $this->end = $end;
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

        $this->rate = $this->start + ($this->end - $this->start) * $fraction;
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
        return "Ramp (start: {$this->start}, end: {$this->end}, steps: {$this->steps})";
    }
}
