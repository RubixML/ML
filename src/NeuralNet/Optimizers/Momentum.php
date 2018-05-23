<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

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
     * @var array
     */
    protected $velocities = [
        //
    ];

    /**
     * @param  float  $rate
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $decay = 0.9)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        if ($decay < 0.0 or $decay > 1.0) {
            throw new InvalidArgumentException('Decay parameter must be a float value between 0 and 1.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
    }

    /**
     * Calculate the step size for each parameter in the network.
     *
     * @param  array  $gradients
     * @return array
     */
    public function step(array $gradients) : array
    {
        $steps = [[[]]];

        foreach ($gradients as $i => $layer) {
            foreach ($layer as $j => $neuron) {
                foreach ($neuron as $k => $gradient) {
                    if (!isset($this->velocities[$i][$j][$k])) {
                        $this->velocities[$i][$j][$k] = 0.0;
                    }

                    $velocity = $this->decay * $this->velocities[$i][$j][$k]
                        + $this->rate * $gradient;

                    $this->velocities[$i][$j][$k] = $velocity;

                    $steps[$i][$j][$k] = $velocity;
                }
            }
        }

        return $steps;
    }
}
