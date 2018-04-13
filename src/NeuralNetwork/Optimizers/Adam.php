<?php

namespace Rubix\Engine\NeuralNetwork\Optimizers;

use Rubix\Engine\NeuralNetwork\Synapse;
use InvalidArgumentException;
use SplObjectStorage;

class Adam implements Optimizer
{
    const EPSILON = 1e-8;

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
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * @param  float  $rate
     * @param  float  $momentumDecay
     * @param  float  $rmsDecay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.01, float $momentumDecay = 0.9, float $rmsDecay = 0.999)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        if ($momentumDecay < 0.0 || $momentumDecay > 1.0) {
            throw new InvalidArgumentException('Momentum decay parameter must be a float between 0 and 1.');
        }

        if ($rmsDecay < 0.0 || $rmsDecay > 1.0) {
            throw new InvalidArgumentException('RMS decay parameter must be a float between 0 and 1.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
        $this->cache = new SplObjectStorage();
    }

    /**
     * Calculate the amount of a step of gradient descent.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Synapse  $synapse
     * @param  float  $gradient
     * @return void
     */
    public function step(Synapse $synapse, float $gradient) : void
    {
        if (!$this->cache->contains($synapse)) {
            $this->cache->attach($synapse, [0.0, 0.0]);
        }

        $this->cache[$synapse] = [
            $this->momentumDecay * $this->cache[$synapse][0] + (1 - $this->momentumDecay) * $gradient,
            $this->rmsDecay * $this->cache[$synapse][1] + (1 - $this->rmsDecay) * $gradient ** 2,
        ];

        $synapse->adjustWeight($this->rate * $this->cache[$synapse][0] / (sqrt($this->cache[$synapse][1]) + self::EPSILON));
    }
}
