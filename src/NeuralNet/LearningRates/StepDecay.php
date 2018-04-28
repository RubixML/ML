<?php

namespace Rubix\Engine\NeuralNet\LearningRates;

use Rubix\Engine\NeuralNet\Synapse;
use InvalidArgumentException;
use SplObjectStorage;

class StepDecay implements LearningRate
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
     * A cache of the sums of squared gradients for each synapse.
     *
     * @var \SplObjectStorage
     */
    protected $steps;

    /**
     * @param  float  $rate
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.01, float $decay = 1e-10)
    {
        if (!$rate > 0.0) {
            throw new InvalidArgumentException('The learning rate must be set to a positive value.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
        $this->steps = new SplObjectStorage();
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
        if (!$this->steps->contains($synapse)) {
            $this->steps->attach($synapse, 0);
        }

        if ($gradient !== 0.0) {
            $this->steps[$synapse] = $this->steps[$synapse] + 1;
        }

        return $this->rate * (1 / (1 + $this->decay * $this->steps[$synapse])) * $gradient;
    }
}
