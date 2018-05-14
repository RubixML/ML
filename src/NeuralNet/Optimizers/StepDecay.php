<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use InvalidArgumentException;

class StepDecay implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The factor to decrease the learning rate by over a period of k steps.
     *
     * @var float
     */
    protected $decay;

    /**
     * The number of steps each parameter has taken. i.e. the number of updates.
     *
     * @var array
     */
    protected $counts = [
        //
    ];

    /**
     * @param  float  $rate
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $decay = 1e-8)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
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
                    if (!isset($this->counts[$i][$j][$k])) {
                        $this->counts[$i][$j][$k] = 0.0;
                    }

                    if ($gradient !== 0.0) {
                        $this->counts[$i][$j][$k]++;
                    }

                    $steps[$i][$j][$k] = $this->rate * $gradient
                        * (1 / (1 + $this->decay * $this->counts[$i][$j][$k]));
                }
            }
        }

        return $steps;
    }
}
