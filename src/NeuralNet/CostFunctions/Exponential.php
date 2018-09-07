<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;
use InvalidArgumentException;

/**
 * Exponential
 *
 * This cost function calculates the exponential of a prediction's squared error
 * thus applying a large penalty to wrong predictions. The resulting gradient of
 * the Exponential loss tends to be steeper than most other cost functions. The
 * magnitude of the error can be scaled by the parameter tau.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Exponential implements CostFunction
{
    /**
     * The scaling parameter i.e. the magnitude of the error to return.
     *
     * @var float
     */
    protected $tau;

    /**
     * @param  float  $tau
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $tau = 1.)
    {
        if ($tau < 0.) {
            throw new InvalidArgumentException('Scaling parameter cannot be'
                . ' less than 0.');
        }

        $this->tau = $tau;
    }

    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0., INF];
    }

    /**
     * Compute the cost.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        return $activations->subtract($expected)->square()
            ->multiplyScalar((1. / $this->tau))->exp()
            ->multiplyScalar($this->tau);
    }

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @param  \Rubix\ML\Other\Structures\Matrix  $delta
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        return $activations->subtract($expected)
            ->multiply($delta)
            ->multiplyScalar(2. / $this->tau);
    }
}
