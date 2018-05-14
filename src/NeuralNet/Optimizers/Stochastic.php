<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use InvalidArgumentException;

class Stochastic implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * @param  float  $rate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        $this->rate = $rate;
    }

    /**
     * Calculate the step size for each parameter in the network.
     *
     * @param  array  $gradients
     * @return array
     */
    public function step(array $gradients) : array
    {
        $steps = [];

        foreach ($gradients as $i => $layer) {
            foreach ($layer as $j => $neuron) {
                foreach ($neuron as $k => $gradient) {
                    $steps[$i][$j][$k] = $this->rate * $gradient;
                }
            }
        }

        return $steps;
    }
}
