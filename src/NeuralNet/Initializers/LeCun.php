<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

use function sqrt;

/**
 * Le Cun
 *
 * Proposed by Yan Le Cun in a paper in 1998, this initializer was one of the
 * first published attempts to control the variance of activations between
 * layers through weight initialization. It remains a good default choice for
 * many hidden layer configurations.
 *
 * References:
 * [1] Y. Le Cun et al. (1998). Efficient Backprop.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeCun implements Initializer
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
        $scale = sqrt(3 / $fanIn);

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
        return 'Le Cun';
    }
}
