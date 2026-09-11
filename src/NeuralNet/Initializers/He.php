<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

use function sqrt;

/**
 * He
 *
 * The He initializer was designed for hidden layers that feed into rectified
 * linear layers such ReLU, Leaky ReLU, ELU, and SELU. It draws from a uniform
 * distribution with limits defined as +/- sqrt(6 / fanIn).
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class He implements Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @internal
     *
     * @param int<0,max> $fanIn
     * @param int<0,max> $fanOut
     * @return Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix
    {
        $scale = sqrt(6.0 / $fanIn);

        return Matrix::uniform($fanOut, $fanIn)->multiply($scale);
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
        return 'He';
    }
}
