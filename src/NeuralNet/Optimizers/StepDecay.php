<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
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
     * @var int
     */
    protected $steps;

    /**
     * @param  float  $rate
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $decay = 1e-8)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
    }

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Network  $network
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        $this->steps = 0;
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Parametric  $layer
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parametric $layer) : Matrix
    {
        $this->steps++;

        return $layer->gradients()
            ->scalarMultiply((1 / (1 + $this->decay * $this->steps)))
            ->scalarMultiply($this->rate);
    }
}
