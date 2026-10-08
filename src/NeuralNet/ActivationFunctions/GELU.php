<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;

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
     * The reciprocal of the square root of 2.
     *
     * @var float
     */
    protected const INV_SQRT2 = M_SQRT1_2;

    /**
     * The reciprocal of the square root of 2 pi i.e. the normalization
     * constant of the standard normal probability density function.
     *
     * @var float
     */
    protected const INV_SQRT_2PI = 0.3989422804014327;

    /**
     * @throws RuntimeException
     */
    public function __construct()
    {
        if (ExtensionIsLoaded::with('tensor')->passes()) {
            ExtensionMinimumVersion::with('tensor', '4.1.0')->check();
        }
    }

    /**
     * Compute the output value.
     *
     * GELU(x) = x Φ(x) = 0.5 x (1 + erf(x / √2))
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix
    {
        return $x->multiply($x->multiplyScalar(self::INV_SQRT2)->erf()->add(1.0))
            ->multiplyScalar(0.5);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * GELU'(x) = Φ(x) + x φ(x)
     *          = 0.5 (1 + erf(x / √2)) + x (2 π)^-0.5 exp(-x² / 2)
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix
    {
        $cdf = $x->multiplyScalar(self::INV_SQRT2)->erf()->add(1.0)
            ->multiplyScalar(0.5);

        $pdf = $x->square()->multiplyScalar(-0.5)->exp()
            ->multiplyScalar(self::INV_SQRT_2PI);

        return $cdf->add($x->multiply($pdf));
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
