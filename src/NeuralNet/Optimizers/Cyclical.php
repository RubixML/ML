<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use InvalidArgumentException;

/**
 * Cyclical
 *
 * The Cyclical optimizer uses a global learning rate that cycles between the
 * lower and upper bound over a designated period while also decaying the
 * uppoer bound by a factor of decay each step. Cyclical learning rates
 * have been shown to help escape local minima and saddle points thus
 * acheiving higher accuracy.
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
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $lower = 0.001,
        float $upper = 0.006,
        int $steps = 2000,
        float $decay = 0.99994
    ) {
        if ($lower <= 0.) {
            throw new InvalidArgumentException('The lower bound must'
                . " be greater than 0, $lower given.");
        }

        if ($lower > $upper) {
            throw new InvalidArgumentException('The lower bound cannot'
                . ' be greater than the upper bound.');
        }

        if ($steps < 1) {
            throw new InvalidArgumentException('The number of steps per'
                . " cycle must be greater than 0, $steps given.");
        }

        if ($decay <= 0. or $decay >= 1.) {
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
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Rubix\Tensor\Tensor $gradient
     */
    public function step(Parameter $param, Tensor $gradient) : void
    {
        $cycle = floor(1 + $this->t / (2 * $this->steps));

        $x = abs($this->t / $this->steps - 2 * $cycle + 1);

        $scale = $this->decay ** $this->t;

        $rate = $this->lower + $this->range * max(0, 1 - $x) * $scale;

        $param->update($gradient->multiply($rate));

        $this->t++;
    }
}
