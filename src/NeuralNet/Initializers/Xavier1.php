<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

use function sqrt;

/**
 * Xavier 1
 *
 * The Xavier 1 initializer draws from a uniform distribution [-limit, limit]
 * where *limit* is squal to sqrt(6 / (fanIn + fanOut)). This initializer is
 * best suited for layers that feed into an activation layer that outputs a
 * value between 0 and 1 such as Softmax or Sigmoid.
 *
 * References:
 * [1] X. Glorot et al. (2010). Understanding the Difficulty of Training Deep
 * Feedforward Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Xavier1 implements Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @internal
     *
     * @param int<0,max> $fanIn
     * @param int<0,max> $fanOut
     * @return \Tensor\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix
    {
        $scale = sqrt(6.0 / ($fanOut + $fanIn));

        return Matrix::uniform($fanOut, $fanIn)
            ->multiply($scale);
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
        return 'Xavier 1';
    }
}
