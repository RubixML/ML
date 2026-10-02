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
        $x3 = $x->square()->multiply($x);

        $inner = $x->add($x3->multiply(self::BETA))
            ->multiplyScalar(self::ALPHA);

        return $x->multiply($inner->tanh()->add(1.0))
            ->multiplyScalar(0.5);
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
        $x3 = $x->square()->multiply($x);

        $alpha = $x3->multiply(0.0356774)->add($x->multiply(self::ALPHA));
        $beta = $x3->multiply(0.0535161)->add($x->multiply(0.398942));

        $tanhA = $alpha->tanh();
        $sech2A = $tanhA->square()->negate()->add(1.0);

        return $tanhA->multiply(0.5)
            ->add($beta->multiply($sech2A))
            ->addScalar(0.5);
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
