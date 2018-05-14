<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use InvalidArgumentException;

class RMSProp implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * @var float
     */
    protected $decay;

    /**
     * A cache of the sums of squared gradients for each synapse.
     *
     * @var array
     */
    protected $cache = [
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

        if ($decay < 0.0 || $decay > 1.0) {
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
                    if (!isset($this->cache[$i][$j][$k])) {
                        $this->cache[$i][$j][$k] = 0.0;
                    }

                    $this->cache[$i][$j][$k] = $this->decay * $this->cache[$i][$j][$k]
                        + (1 - $this->decay)
                        * $gradient ** 2;

                    $steps[$i][$j][$k] = $this->rate * $gradient
                        / (sqrt($this->cache[$i][$j][$k]) + self::EPSILON);
                }
            }
        }

        return $steps;
    }
}
