<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use InvalidArgumentException;

/**
 * Swish
 *
 * Swish is a smooth and non-monotonic rectified activation function. The inputs are weighted by the [Sigmoid](sigmoid.md)
 * activation function acting as a self-gating mechanism. In addition, the `beta` parameter allows you to adjust the gate
 * such that you can interpolate between the ReLU function and the linear function as `beta` goes from 0 to infinity.
 *
 * References:
 * [1] S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function
 * Approximation in Reinforcement Learning.
 * [2] P. Ramachandran et al. (2017). Swish: A Self-gated Activation Function.
 * [3] P. Ramachandran et al. (2017). Searching for Activation Functions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Swish implements ActivationFunction
{
    /**
     * The parameter that adjusts the slope of the sigmoid gating mechanism.
     *
     * @var float
     */
    protected float $beta;

    /**
     * The sigmoid activation function.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid
     */
    protected \Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid $sigmoid;

    /**
     * @param float $beta
     * @throws \InvalidArgumentException
     */
    public function __construct(float $beta = 1.0)
    {
        if ($beta < 0.0) {
            throw new InvalidArgumentException('Beta must be greater than'
                . " 0, $beta given.");
        }

        $this->beta = $beta;
        $this->sigmoid = new Sigmoid();
    }

    /**
     * Compute the output value.
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $this->sigmoid->compute($z->multiply($this->beta))
            ->multiply($z);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        $ones = Matrix::ones(...$computed->shape());

        return $computed->divide($z)
            ->multiply($ones->subtract($computed))
            ->add($computed);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Swish (beta: {$this->beta})";
    }
}
