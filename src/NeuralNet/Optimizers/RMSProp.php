<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

/**
 * RMS Prop
 *
 * An adaptive gradient technique that divides the current gradient over a
 * rolling window of magnitudes of recent gradients.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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
     * @var \MathPHP\LinearAlgebra\Matrix
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
    }

    /**
     * Initialize the layer optimizer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function initialize(Matrix $weights) : void
    {
        $this->cache = MatrixFactory::zero($weights->getM(),
            $weights->getN());
    }

    /**
     * Calculate a gradient descent step for a layer given a matrix of gradients.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Matrix $gradients) : Matrix
    {
        $this->cache = $this->cache->scalarMultiply($this->decay)
            ->add($gradients->hadamardProduct($gradients->scalarMultiply(1 - $this->decay)));

        $steps = [[]];

        foreach ($gradients->getMatrix() as $i => $row) {
            foreach ($row as $j => $column) {
                $steps[$i][$j] = $this->rate * $column
                    / (sqrt($this->cache[$i][$j]) + self::EPSILON);
            }
        }

        return new Matrix($steps);
    }
}
