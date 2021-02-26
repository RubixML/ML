<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function exp;

/**
 * ELU
 *
 * Exponential Linear Units are a type of rectifier that soften the transition
 * from non-activated to activated using the exponential function.
 *
 * References:
 * [1] D. A. Clevert et al. (2016). Fast and Accurate Deep Network Learning by
 * Exponential Linear Units.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ELU implements ActivationFunction
{
    /**
     * At which negative value the ELU will saturate. For example if alpha
     * equals 1, the leaked value will never be greater than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param float $alpha
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha cannot be less than'
                . " 0, $alpha given.");
        }

        $this->alpha = $alpha;
    }

    /**
     * Compute the output value.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0.0 ? $z : $this->alpha * (exp($z) - 1.0);
    }

    /**
     * @internal
     *
     * @param float $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed > 0.0 ? 1.0 : $computed + $this->alpha;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "ELU (alpha: {$this->alpha})";
    }
}
