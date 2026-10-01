<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * SELU
 *
 * Scaled Exponential Linear Unit is a self-normalizing activation function
 * based on the ELU activation function. Neuronal activations of SELU networks
 * automatically converge toward zero mean and unit variance, unlike explicitly
 * normalized networks such as those with [Batch Norm](#batch-norm).
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
    /**
     * The value at which leakage starts to saturate.
     *
     * @var float
     */
    public const float ALPHA = 1.6732632423543772848170429916717;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    public const float SCALE = 1.0507009873554804934193349852946;

    /**
     * The scaling coefficient multiplied by alpha.
     *
     * @var float
     */
    protected const float BETA = self::SCALE * self::ALPHA;

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
        return $x->greater(0.0)
            ->multiply($x->multiply(self::SCALE))
            ->add($x->lessEqual(0.0)
            ->multiply($x->expm1())
            ->multiply(self::BETA));
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
        $zHat = $z->lessEqual(0.0)->multiply($z->add(self::BETA));

        return $z->greater(0.0)->multiply(self::SCALE)->add($zHat);
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
        return 'SELU';
    }
}
