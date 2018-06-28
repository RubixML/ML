<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Layers\Parametric;
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
     * The size of every floor in steps. i.e. the number of steps to take before
     * applying another factor of decay.
     *
     * @var int
     */
    protected $k;

    /**
     * The factor to decrease the learning rate by over a period of k steps.
     *
     * @var float
     */
    protected $decay;

    /**
     * The number of steps each parameter has taken until the next floor.
     *
     * @var int
     */
    protected $steps;

    /**
     * The number of floors passed.
     *
     * @var int
     */
    protected $floors;

    /**
     * @param  float  $rate
     * @param  int  $k
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, int $k = 10, float $decay = 1e-5)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        $this->rate = $rate;
        $this->k = $k;
        $this->decay = $decay;
    }

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        $this->steps = 0;
        $this->floors = 0;
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return float
     */
    public function step(Parametric $layer) : float
    {
        $this->steps++;

        if ($this->steps > $this->k) {
            $this->floors++;

            $this->steps = 0;
        }

        $rate = $this->rate * (1 / (1 + $this->decay * $this->floors));

        $steps = $layer->gradients()->scalarMultiply($rate);

        $layer->update($steps);

        return $steps->oneNorm();
    }
}
