<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Layers\Parametric;
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
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        $this->rate = $rate;
    }

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        //
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return float
     */
    public function step(Parametric $layer) : float
    {
        $steps = $layer->gradients()->scalarMultiply($this->rate);

        $layer->update($steps);

        return $steps->oneNorm();
    }
}
