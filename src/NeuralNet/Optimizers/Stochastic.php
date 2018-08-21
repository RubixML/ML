<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

/**
 * Stochastic
 *
 * A constant learning rate gradient descent optimizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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
            throw new InvalidArgumentException('The learning rate must be'
                . ' greater than 0.');
        }

        $this->rate = $rate;
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradients) : Matrix
    {
        return $gradients->scalarMultiply($this->rate);
    }
}
