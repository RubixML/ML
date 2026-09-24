<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * GELU
 *
 * Gaussian Error Linear Units (GELUs) are rectifiers that are gated by the magnitude of their input rather
 * than the sign of their input as with ReLU variants. Their output can be interpreted as the expected value
 * of a neuron with random dropout regularization applied.
 *
 * [1] D. Hendrycks et al. (2018). Gaussian Error Linear Units (GELUs).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GELU implements ActivationFunction
{
    /**
     * The square root of two over pi.
     *
     * @var float
     */
    protected const ALPHA = 0.7978845608;

    /**
     * Gaussian error function approximation term.
     *
     * @var float
     */
    protected const BETA = 0.044715;

    /**
     * Compute the output value.
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix
    {
        $inner = $x->add($x->pow(3.0)->multiply(self::BETA))
            ->multiply(self::ALPHA);

        return $x->multiply($this->tanh($inner)->add(1.0))
            ->multiply(0.5);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix
    {
        $xHat = $x->pow(3.0);

        $alpha = $xHat->multiply(0.0356774)->add($x->multiply(self::ALPHA));
        $beta = $xHat->multiply(0.0535161)->add($x->multiply(0.398942));

        $tanhA = $this->tanh($alpha);
        $sech2A = $tanhA->pow(2.0)->negate()->add(1.0);

        return $tanhA->multiply(0.5)
            ->add($beta->multiply($sech2A))
            ->add(0.5);
    }

    /**
     * Compute the elementwise hyperbolic tangent.
     *
     * @internal
     *
     * @param Matrix $x
     * @return Matrix
     */
    protected function tanh(Matrix $x) : Matrix
    {
        return $x->multiply(2.0)
            ->exp()
            ->add(1.0)
            ->reciprocal()
            ->multiply(-2.0)
            ->add(1.0);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'GELU';
    }
}
