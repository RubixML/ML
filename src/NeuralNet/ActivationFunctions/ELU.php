<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
     * @throws InvalidArgumentException
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
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix
    {
        return $x->greater(0.0)->multiply($x)
            ->add($x->lessEqual(0.0)
            ->multiply($x->expm1())
            ->multiplyScalar($this->alpha));
    }

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix
    {
        $zHat = $z->lessEqual(0.0)->multiply($z->add($this->alpha));

        return $z->greater(0.0)->add($zHat);
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
