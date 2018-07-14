<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

/**
 * Step Decay
 *
 * A learning rate decay stochastic optimizer that reduces the learning rate by
 * a factor of the decay parameter when it reaches a new floor (takes k steps).
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
    protected $k;

    /**
     * The factor to decrease the learning rate by over a period of k steps.
     *
     * @var float
     */
    protected $decay;

    /**
     * The number of steps each parameter has taken until the next floor.
     *
     * @var int
     */
    protected $steps;

    /**
     * The number of floors passed.
     *
     * @var int
     */
    protected $floors;

    /**
     * @param  float  $rate
     * @param  int  $k
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, int $k = 10, float $decay = 1e-5)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        $this->rate = $rate;
        $this->k = $k;
        $this->decay = $decay;
    }

    /**
     * Initialize the layer optimizer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function initialize(Matrix $weights) : void
    {
        $this->steps = 0;
        $this->floors = 0;
    }

    /**
     * Calculate a gradient descent step for a layer given a matrix of gradients.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Matrix $gradients) : Matrix
    {
        $this->steps++;

        if ($this->steps > $this->k) {
            $this->steps = 0;
            $this->floors++;
        }

        $rate = $this->rate * (1 / (1 + $this->decay * $this->floors));

        return $gradients->scalarMultiply($rate);
    }
}
