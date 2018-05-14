<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use InvalidArgumentException;

class AdaGrad implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * A cache of the sum of the squared gradients for each paramter.
     *
     * @var array
     */
    protected $cache = [
        //
    ];

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
        $steps = [[[]]];

        foreach ($gradients as $i => $layer) {
            foreach ($layer as $j => $neuron) {
                foreach ($neuron as $k => $gradient) {
                    if (!isset($this->cache[$i][$j][$k])) {
                        $this->cache[$i][$j][$k] = 0.0;
                    }

                    $this->cache[$i][$j][$k] += $gradient ** 2;

                    $steps[$i][$j][$k] =  $this->rate * $gradient
                        / (sqrt($this->cache[$i][$j][$k]) + self::EPSILON);
                }
            }
        }

        return $steps;
    }
}
