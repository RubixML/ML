<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
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
class SELU implements Rectifier
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
                . ' positive.');
        }

        if ($scale < 1.) {
            throw new InvalidArgumentException('Scale must be greater than 1.');
        }

        $this->scale = $scale;
        $this->alpha = $alpha;
        $this->beta = $scale * $alpha;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [-$this->beta, INF];
    }

    /**
     * Compute the output value.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map(function ($value) {
            return $value > 0.
                ? $this->scale * $value
                : $this->beta * exp($value) - $this->alpha;
        });
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @param  \MathPHP\LinearAlgebra\Matrix  $computed
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map(function ($activation) {
            return $activation > 0.
                ? $this->scale
                : $this->scale * ($activation + $this->alpha);
        });
    }
}
