<?php

namespace Rubix\ML\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Cyclical
 *
 * The Cyclical scheduler cycles the learning rate between the lower and upper
 * bound over a designated period while also decaying the upper bound by the
 * decay coefficient at each step. Cyclical learning rates have been shown to
 * help escape bad local minima and saddle points thus achieving lower
 * training loss.
 *
 * References:
 * [1] L. N. Smith. (2017). Cyclical Learning Rates for Training Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cyclical implements Scheduler
{
    /**
     * The lower bound on the learning rate.
     *
     * @var float
     */
    protected float $lower;

    /**
     * The upper bound on the learning rate.
     *
     * @var float
     */
    protected float $upper;

    /**
     * The range of the learning rate.
     *
     * @var float
     */
    protected float $range;

    /**
     * The number of steps in every cycle.
     *
     * @var int
     */
    protected int $length;

    /**
     * The exponential scaling factor applied to each step as decay.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The precomputed learning rate for the current step.
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
     * @param float $lower
     * @param float $upper
     * @param int $length
     * @param float $decay
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $lower = 0.001,
        float $upper = 0.006,
        int $length = 2000,
        float $decay = 0.99994
    ) {
        if ($lower <= 0.0) {
            throw new InvalidArgumentException('Lower bound must be'
                . " greater than 0, $lower given.");
        }

        if ($lower > $upper) {
            throw new InvalidArgumentException('Lower bound cannot be'
                . ' greater than the upper bound.');
        }

        if ($length < 1) {
            throw new InvalidArgumentException('The cycle length must be'
                . " greater than 0, $length given.");
        }

        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->lower = $lower;
        $this->upper = $upper;
        $this->range = $upper - $lower;
        $this->length = $length;
        $this->decay = $decay;
        $this->rate = $lower;
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

        $cycle = floor(1 + $this->t / (2 * $this->length));

        $x = abs($this->t / $this->length - 2 * $cycle + 1);

        $scale = $this->decay ** $this->t;

        $rate = $this->lower + $this->range * max(0, 1 - $x) * $scale;

        $this->rate = $rate;
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
        return "Cyclical (lower: {$this->lower}, upper: {$this->upper},"
            . " length: {$this->length}, decay: {$this->decay})";
    }
}
