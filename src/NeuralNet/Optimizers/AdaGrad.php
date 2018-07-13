<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

class AdaGrad implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The sum of squared gradient matrices for each layer.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $cache;

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
     * Initialize the layer optimizer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function initialize(Matrix $weights) : void
    {
        $this->cache = MatrixFactory::zero($weights->getM(), $weights->getN());
    }

    /**
     * Calculate a gradient descent step for a layer given a matrix of gradients.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Matrix $gradients) : Matrix
    {
        $this->cache = $this->cache->add($gradients->hadamardProduct($gradients));

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
