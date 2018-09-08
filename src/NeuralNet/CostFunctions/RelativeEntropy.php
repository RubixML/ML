<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;
use InvalidArgumentException;

/**
 * Relative Entropy
 *
 * Relative Entropy or *Kullback-Leibler divergence* is a measure of how the
 * expectation and activation of the network diverge.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RelativeEntropy implements CostFunction
{
    /**
     * The derivative smoothing parameter i.e a small value to add to the
     * denominator of the derivative calculation for numerical stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $epsilon = 1e-20)
    {
        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Epsilon must be greater than'
                . ' 0.');
        }

        $this->epsilon = $epsilon;
    }

    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array
    {
        return [-INF, INF];
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
        $expected = $expected->clip($this->epsilon, 1.);
        $activations = $activations->clip($this->epsilon, 1.);

        return $expected->multiply($expected->divide($activations)->log());
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
        $expected = $expected->clip($this->epsilon, 1.);
        $activations = $activations->clip($this->epsilon, 1.);

        return $activations->subtract($expected)
            ->divide($activations);
    }
}
