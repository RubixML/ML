<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use Rubix\Engine\NeuralNet\Synapse;
use InvalidArgumentException;
use SplObjectStorage;

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
     * @var \SplObjectStorage
     */
    protected $cache;

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
        $this->cache = new SplObjectStorage();
    }

    /**
     * Calculate the amount of a step of gradient descent.
     *
     * @param  \Rubix\Engine\NeuralNet\Synapse  $synapse
     * @param  float  $gradient
     * @return float
     */
    public function step(Synapse $synapse, float $gradient) : float
    {
        if (!$this->cache->contains($synapse)) {
            $this->cache->attach($synapse, 0.0);
        }

        $this->cache[$synapse] = $this->decay * $this->cache[$synapse] + (1 - $this->decay) * $gradient ** 2;

        return $this->rate * $gradient / (sqrt($this->cache[$synapse]) + self::EPSILON);
    }
}
