<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * Momentum
 *
 * Momentum adds velocity to each step until exhausted. It does so by
 * accumulating momentum from past updates and adding a factor to the current
 * step.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Momentum implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The rate at which the momentum force decays.
     *
     * @var float
     */
    protected $decay;

    /**
     * The memoized velocity matrices.
     *
     * @var \SplObjectStorage
     */
    protected $velocities;

    /**
     * @param  float  $rate
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $decay = 0.9)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . ' greater than 0.');
        }

        if ($decay < 0. or $decay > 1.) {
            throw new InvalidArgumentException('Velocity decay must be between'
                . ' 0 and 1.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
        $this->velocities = new SplObjectStorage();
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \Rubix\ML\Other\Structures\Matrix  $gradients
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradients) : Matrix
    {
        if ($this->velocities->contains($parameter)) {
            $velocities = $this->velocities[$parameter];
        } else {
            $velocities = Matrix::zeros(...$parameter->w()->shape());

            $this->velocities->attach($parameter, $velocities);
        }

        $velocities = $gradients
            ->scalarMultiply($this->rate)
            ->add($velocities->scalarMultiply($this->decay));

        $this->velocities[$parameter] = $velocities;

        return $velocities;
    }
}
