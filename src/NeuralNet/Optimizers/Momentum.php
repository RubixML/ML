<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Layers\Parametric;
use InvalidArgumentException;
use SplObjectStorage;

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
        $this->velocities = new SplObjectStorage();
    }

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\ML\NeuralNet\Network  $network
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        $this->velocities->attach($layer, MatrixFactory::zero($layer->weights()
            ->getM(), $layer->weights()->getN()));
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parametric $layer) : Matrix
    {
        $velocities = $layer->gradients()
            ->add($this->velocities[$layer]->scalarMultiply($this->decay))
            ->scalarMultiply($this->rate);

        $this->velocities[$layer] = $velocities;

        return $velocities;
    }
}
