<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\Engine\NeuralNet\Layers\Parametric;
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
     * @param  \Rubix\Engine\NeuralNet\Network  $network
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        //
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Parametric  $layer
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parametric $layer) : Matrix
    {
        return $layer->gradients()->scalarMultiply($this->rate);
    }
}
