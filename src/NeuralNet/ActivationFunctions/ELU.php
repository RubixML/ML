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
    protected float $alpha;

    /**
     * @param float $alpha
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be greater than'
                . " 0, $alpha given.");
        }

        $this->alpha = $alpha;
    }

    /**
     * Compute the activation.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function activate(Matrix $input) : Matrix
    {
        return $input->map([$this, '_activate']);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $output
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $input, Matrix $output) : Matrix
    {
        return $output->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _activate(float $input) : float
    {
        return $input > 0.0
            ? $input
            : $this->alpha * (exp($input) - 1.0);
    }

    /**
     * @internal
     *
     * @param float $output
     * @return float
     */
    public function _differentiate(float $output) : float
    {
        return $output > 0.0
            ? 1.0
            : $output + $this->alpha;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "ELU (alpha: {$this->alpha})";
    }
}
