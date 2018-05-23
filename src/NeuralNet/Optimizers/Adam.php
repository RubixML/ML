<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use InvalidArgumentException;

class Adam implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The decay rate of the momentum.
     *
     * @var float
     */
    protected $momentumDecay;

    /**
     * The decay rate of the RMS property.
     *
     * @var float
     */
    protected $rmsDecay;

    /**
     * A cache of the current rms and velocities of each synapse.
     *
     * @var array
     */
    protected $cache = [
        //
    ];

    /**
     * @param  float  $rate
     * @param  float  $momentumDecay
     * @param  float  $rmsDecay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.9, float $rmsDecay = 0.999)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        if ($momentumDecay < 0.0 or $momentumDecay > 1.0) {
            throw new InvalidArgumentException('Momentum decay parameter must be a float between 0 and 1.');
        }

        if ($rmsDecay < 0.0 or $rmsDecay > 1.0) {
            throw new InvalidArgumentException('RMS decay parameter must be a float between 0 and 1.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
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
                        $this->cache[$i][$j][$k] = [0.0, 0.0];
                    }

                    $this->cache[$i][$j][$k] = [
                        $this->momentumDecay * $this->cache[$i][$j][$k][0]
                            + (1 - $this->momentumDecay) * $gradient,

                        $this->rmsDecay * $this->cache[$i][$j][$k][1]
                            + (1 - $this->rmsDecay) * $gradient ** 2,
                    ];

                    $steps[$i][$j][$k] = $this->rate * $this->cache[$i][$j][$k][0]
                        / (sqrt($this->cache[$i][$j][$k][1]) + self::EPSILON);
                }
            }
        }

        return $steps;
    }
}
