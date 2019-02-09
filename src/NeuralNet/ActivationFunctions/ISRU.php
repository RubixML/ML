<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

/**
 * ISRU
 *
 * Inverse Square Root units have a curve similar to Hyperbolic Tangent and
 * Sigmoid but use the inverse of the square root function instead. It is
 * purported by the authors to be computationally less complex than either of
 * the aforementioned. In addition, ISRU allows the parameter alpha to control
 * the range of activation such that it equals + or - 1 / sqrt(alpha).
 *
 * References:
 * [1] B. Carlile et al. (2017). Improving Deep Learning by Inverse Square Root
 * Linear Units.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ISRU implements ActivationFunction
{
    /**
     * At which point the output values of the function will saturdate. i.e.
     * alpha = 2.means that the output will be between + or - 1 / sqrt(2.).
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param float $alpha
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 1.)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('Alpha parameter must be'
                . ' positive.');
        }

        $this->alpha = $alpha;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        $r = 1. / sqrt($this->alpha);

        return [-$r, $r];
    }

    /**
     * Compute the output value.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @param \Rubix\Tensor\Matrix $computed
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $z->map([$this, '_differentiate']);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z / sqrt(1. + $this->alpha * $z ** 2);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _differentiate(float $z) : float
    {
        return (1. / sqrt((1. + $this->alpha * $z ** 2))) ** 3;
    }
}
