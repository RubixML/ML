<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\Engine\NeuralNet\Layers\Parametric;
use InvalidArgumentException;
use SplObjectStorage;

class Adam implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The decay rate of the momentum property.
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
     * The RMS matrices for each layer.
     *
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * The velocity matrices per layer.
     *
     * @var \SplObjectStorage
     */
    protected $velocities;

    /**
     * @param  float  $rate
     * @param  float  $momentumDecay
     * @param  float  $rmsDecay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.9, float $rmsDecay = 0.999)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        if ($momentumDecay < 0.0 or $momentumDecay > 1.0) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . ' 0 and 1.');
        }

        if ($rmsDecay < 0.0 or $rmsDecay > 1.0) {
            throw new InvalidArgumentException('RMS decay rate must be between'
                . ' 0 and 1.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
        $this->velocities = new SplObjectStorage();
        $this->cache = new SplObjectStorage();
    }

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Parametric  $layer
     * @return void
     */
    public function initialize(Parametric $layer) : void
    {
        $this->velocities->attach($layer, MatrixFactory::zero($layer->weights()
            ->getM(), $layer->weights()->getN()));

        $this->cache->attach($layer, MatrixFactory::zero($layer->weights()
            ->getM(), $layer->weights()->getN()));
    }

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Parametric  $layer
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parametric $layer) : Matrix
    {
        $velocities = $this->velocities[$layer]
            ->scalarMultiply($this->momentumDecay)
            ->add($layer->gradients()
                ->scalarMultiply(1 - $this->momentumDecay));

        $cache = $this->cache[$layer]
            ->scalarMultiply($this->rmsDecay)
            ->add($layer->gradients()
                ->hadamardProduct($layer->gradients()
                ->scalarMultiply(1 - $this->rmsDecay)));

        $steps = [];

        foreach ($layer->gradients()->getMatrix() as $i => $row) {
            foreach ($row as $j => $column) {
                $steps[$i][$j] = $this->rate * $velocities[$i][$j]
                    / (sqrt($cache[$i][$j]) + self::EPSILON);
            }
        }

        $this->velocities[$layer] = $velocities;
        $this->cache[$layer] = $cache;

        return new Matrix($steps);
    }
}
