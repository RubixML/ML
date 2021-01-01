<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
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
 */
class Cyclical implements Optimizer
{
    /**
     * The lower bound on the learning rate.
     *
     * @var float
     */
    protected $lower;

    /**
     * The upper bound on the learning rate.
     *
     * @var float
     */
    protected $upper;

    /**
     * The range of the learning rate.
     *
     * @var float
     */
    protected $range;

    /**
     * The number of steps in every cycle.
     *
     * @var int
     */
    protected $steps;

    /**
     * The exponential scaling factor applied to each step as decay.
     *
     * @var float
     */
    protected $decay;

    /**
     * The number of steps taken so far.
     *
     * @var int
     */
    protected $t = 0;

    /**
     * @param float $lower
     * @param float $upper
     * @param int $steps
     * @param float $decay
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        float $lower = 0.001,
        float $upper = 0.006,
        int $steps = 2000,
        float $decay = 0.99994
    ) {
        if ($lower <= 0.0) {
            throw new InvalidArgumentException('Lower bound must be'
                . " greater than 0, $lower given.");
        }

        if ($lower > $upper) {
            throw new InvalidArgumentException('Lower bound cannot be'
                . ' reater than the upper bound.');
        }

        if ($steps < 1) {
            throw new InvalidArgumentException('The number of steps per'
                . " cycle must be greater than 0, $steps given.");
        }

        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->lower = $lower;
        $this->upper = $upper;
        $this->range = $upper - $lower;
        $this->steps = $steps;
        $this->decay = $decay;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        $cycle = floor(1 + $this->t / (2 * $this->steps));

        $x = abs($this->t / $this->steps - 2 * $cycle + 1);

        $scale = $this->decay ** $this->t;

        $rate = $this->lower + $this->range * max(0, 1 - $x) * $scale;

        ++$this->t;

        return $gradient->multiply($rate);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Cyclical (lower: {$this->lower}, upper: {$this->upper},"
            . " steps: {$this->steps}, decay: {$this->decay})";
    }
}
