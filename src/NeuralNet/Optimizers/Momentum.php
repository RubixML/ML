<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

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
     * A table storing the current velocity of each parameter.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
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
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        if ($decay < 0.0 or $decay > 1.0) {
            throw new InvalidArgumentException('Velocity decay must be between'
                . ' 0 and 1.');
        }

        $this->rate = $rate;
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
        $this->velocities = MatrixFactory::zero($weights->getM(),
            $weights->getN());
    }

    /**
     * Calculate a gradient descent step for a layer given a matrix of gradients.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Matrix $gradients) : Matrix
    {
        $this->velocities = $gradients
            ->add($this->velocities->scalarMultiply($this->decay))
            ->scalarMultiply($this->rate);

        return $this->velocities;
    }
}
