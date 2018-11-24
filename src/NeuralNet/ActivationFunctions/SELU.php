<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

/**
 * SELU
 *
 * Scaled Exponential Linear Unit is a self-normalizing activation function
 * based on the ELU activation function.
 *
 * References:
 * [1] G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SELU implements ActivationFunction
{
    const ALPHA = 1.6732632423543772848170429916717;
    const SCALE = 1.0507009873554804934193349852946;

    /**
     * At which negative value the SELU will saturate. i.e. alpha = 1.means
     * that the leakage will never be more than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The scaling factor.
     *
     * @var float
     */
    protected $scale;

    /**
     * The exponential leakage coefficient.
     * 
     * @var float
     */
    protected $beta;

    /**
     * @param  float  $scale
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = self::ALPHA, float $scale = self::SCALE)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('Alpha parameter must be'
                . " greater than 0, $alpha given.");
        }

        if ($scale < 1.) {
            throw new InvalidArgumentException('Scale must be greater than'
                . " 1, $scale given.");
        }

        $this->scale = $scale;
        $this->alpha = $alpha;
        $this->beta = $scale * $alpha;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-$this->beta, INF];
    }

    /**
     * Compute the output value.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @param  \Rubix\Tensor\Matrix  $computed
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param  float  $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0. ? $this->scale * $z : $this->beta * (exp($z) - 1.);
    }

    /**
     * @param  float  $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed > 0. ? $this->scale : $computed + 1.;
    }
}
