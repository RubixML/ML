<?php

namespace Rubix\ML\NeuralNet\Optimizers\Cyclical;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Cyclical
 *
 * The Cyclical optimizer uses a global learning rate that cycles between the
 * lower and upper bound over a designated period while also decaying the
 * upper bound by the decay coefficient at each step. Cyclical learning rates
 * have been shown to help escape bad local minima and saddle points thus
 * achieving lower training loss.
 *
 * References:
 * [1] L. N. Smith. (2017). Cyclical Learning Rates for Training Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Cyclical implements Optimizer
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
    protected int $losses;

    /**
     * The exponential scaling factor applied to each step as decay.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The number of steps taken so far.
     *
     * @var int
     */
    protected int $t = 0;

    /**
     * @param float $lower
     * @param float $upper
     * @param int $losses
     * @param float $decay
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $lower = 0.001,
        float $upper = 0.006,
        int $losses = 2000,
        float $decay = 0.99994
    ) {
        if ($lower <= 0.0) {
            throw new InvalidArgumentException(
                "Lower bound must be greater than 0, $lower given."
            );
        }

        if ($lower > $upper) {
            throw new InvalidArgumentException(
                'Lower bound cannot be reater than the upper bound.'
            );
        }

        if ($losses < 1) {
            throw new InvalidArgumentException(
                "The number of steps per cycle must be greater than 0, $losses given."
            );
        }

        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException(
                "Decay must be between 0 and 1, $decay given."
            );
        }

        $this->lower = $lower;
        $this->upper = $upper;
        $this->range = $upper - $lower;
        $this->losses = $losses;
        $this->decay = $decay;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * Cyclical learning rate schedule (per-step, element-wise update):
     *   - Cycle index:           cycle = floor(1 + t / (2 · losses))
     *   - Triangular position:   x     = | t / losses − 2 · cycle + 1 |
     *   - Exponential decay:     scale = decay^t
     *   - Learning rate at t:    η_t   = lower + (upper − lower) · max(0, 1 − x) · scale
     *   - Returned step:         Δθ_t  = η_t · g_t
     *
     * where:
     *   - t is the current step counter (incremented after computing η_t),
     *   - losses is the number of steps per cycle,
     *   - lower and upper are the learning rate bounds,
     *   - decay is the multiplicative decay applied each step,
     *   - g_t is the current gradient.
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray
    {
        $cycle = floor(1 + $this->t / (2 * $this->losses));

        $x = abs($this->t / $this->losses - 2 * $cycle + 1);

        $scale = $this->decay ** $this->t;

        $rate = $this->lower + $this->range * max(0, 1 - $x) * $scale;

        ++$this->t;

        return NumPower::multiply($gradient, $rate);
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
            . " steps: {$this->losses}, decay: {$this->decay})";
    }
}
