<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

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
     * @param  float  $expected
     * @param  float  $activation
     * @return float
     */
    public function compute(float $expected, float $activation) : float
    {
        return $this->tau * M_E ** ((1. / $this->tau) * ($activation - $expected) ** 2);
    }

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  float  $expected
     * @param  float  $activation
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $expected, float $activation, float $computed) : float
    {
        return (2. / $this->tau) * ($activation - $expected) * $computed;
    }
}
