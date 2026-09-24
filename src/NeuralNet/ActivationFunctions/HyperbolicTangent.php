<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * Hyperbolic Tangent
 *
 * S-shaped function that squeezes the input value into an output space between
 * -1 and 1 centered at 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HyperbolicTangent implements ActivationFunction
{
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
        return $this->tanh($x);
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
        return $z->pow(2.0)->negate()->add(1.0);
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Hyperbolic Tangent';
    }
}
