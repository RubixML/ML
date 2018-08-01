<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * Step Decay
 *
 * A learning rate decay stochastic optimizer that reduces the learning rate by
 * a factor of the decay parameter when it reaches a new floor. The number of
 * steps needed to reach a new floor is given by the steps parameter.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class StepDecay implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The size of every floor in steps. i.e. the number of steps to take before
     * applying another factor of decay.
     *
     * @var int
     */
    protected $steps;

    /**
     * The factor to decrease the learning rate by over a period of k steps.
     *
     * @var float
     */
    protected $decay;

    /**
     * The number of steps each parameter has taken.
     *
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * @param  float  $rate
     * @param  int  $steps
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, int $steps = 10, float $decay = 1e-5)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        if ($steps < 1) {
            throw new InvalidArgumentException('The number of steps per floor'
                . ' must be greater than 0.');
        }

        if ($decay < 0) {
            throw new InvalidArgumentException('The decay rate must be'
                . ' positive');
        }

        $this->rate = $rate;
        $this->steps = $steps;
        $this->decay = $decay;
        $this->cache = new SplObjectStorage();
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradients) : Matrix
    {
        if ($this->cache->contains($parameter)) {
            $steps = $this->cache[$parameter];
        } else {
            $steps = 0;

            $this->cache->attach($parameter, $steps);
        }

        $steps++;

        $rate = $this->rate * (1 / (1 + $this->decay * ($steps / $this->steps)));

        $this->cache[$parameter] = $steps;

        return $gradients->scalarMultiply($rate);
    }
}
