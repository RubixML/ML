<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
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
     * @return float[]
     */
    public function range() : array
    {
        return [-INF, INF];
    }

    /**
     * Compute the cost.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        $expected = $expected->clip($this->epsilon, 1.);
        $activations = $activations->clip($this->epsilon, 1.);

        return $expected->multiply($expected->divide($activations)->log());
    }

    /**
     * Calculate the derivatives of the cost function with respect to the
     * output activation.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @param  \Rubix\Tensor\Matrix  $delta
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        $expected = $expected->clip($this->epsilon, 1.);
        $activations = $activations->clip($this->epsilon, 1.);

        return $activations->subtract($expected)
            ->divide($activations);
    }
}
