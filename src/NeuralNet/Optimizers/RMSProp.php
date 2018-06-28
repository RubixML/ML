<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Layers\Parametric;
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
     * The decay rate parameter.
     *
     * @var float
     */
    protected $decay;

    /**
     * A cache of the sums of squared gradients for each layer.
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
    public function __construct(float $rate = 0.001, float $decay = 0.9)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        if ($decay < 0.0 or $decay > 1.0) {
            throw new InvalidArgumentException('Decay rate must be between 0'
                . ' and 1.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
        $this->cache = new SplObjectStorage();
    }

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        $this->cache->attach($layer, MatrixFactory::zero($layer->weights()
            ->getM(), $layer->weights()->getN()));
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return float
     */
    public function step(Parametric $layer) : float
    {
        $cache = $this->cache[$layer]
            ->scalarMultiply($this->decay)
            ->add($layer->gradients()
                ->hadamardProduct($layer->gradients()
                ->scalarMultiply(1 - $this->decay)));

        $steps = [[]];

        foreach ($layer->gradients()->getMatrix() as $i => $row) {
            foreach ($row as $j => $column) {
                $steps[$i][$j] = $this->rate * $column
                    / (sqrt($cache[$i][$j]) + self::EPSILON);
            }
        }

        $this->cache[$layer] = $cache;

        $steps = new Matrix($steps);

        $layer->update($steps);

        return $steps->oneNorm();
    }
}
