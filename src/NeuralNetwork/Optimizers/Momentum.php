<?php

namespace Rubix\Engine\NeuralNetwork\Optimizers;

use Rubix\Engine\NeuralNetwork\Synapse;
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
    public function __construct(float $rate = 0.01, float $decay = 0.9)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        if ($decay < 0.0 || $decay > 1.0) {
            throw new InvalidArgumentException('Decay parameter must be a float value between 0 and 1.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
        $this->velocities = new SplObjectStorage();
    }

    /**
     * Calculate the amount of a step of gradient descent.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Synapse  $synapse
     * @param  float  $gradient
     * @return float
     */
    public function step(Synapse $synapse, float $gradient) : float
    {
        if (!$this->velocities->contains($synapse)) {
            $this->velocities->attach($synapse, 0.0);
        }

        $velocity = $this->decay * $this->velocities[$synapse] + $this->rate * $gradient;

        $this->velocities[$synapse] = $velocity;

        return $velocity;
    }
}
