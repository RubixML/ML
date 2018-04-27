<?php

namespace Rubix\Engine\NeuralNetwork\Optimizers;

use Rubix\Engine\NeuralNetwork\Synapse;
use InvalidArgumentException;
use SplObjectStorage;

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
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * @param  float  $rate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.01)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        $this->rate = $rate;
        $this->cache = new SplObjectStorage();
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
        if (!$this->cache->contains($synapse)) {
            $this->cache->attach($synapse, 0.0);
        }

        $this->cache[$synapse] += $gradient ** 2;

        return $this->rate * $gradient / (sqrt($this->cache[$synapse]) + self::EPSILON);
    }
}
