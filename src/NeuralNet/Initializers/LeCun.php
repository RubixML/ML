<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Rubix\Tensor\Matrix;

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
     * @param  int  $fanIn
     * @param  int  $fanOut
     * @return \Rubix\Tensor\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix
    {
        $scale = sqrt(3 / $fanIn);

        return Matrix::uniform($fanOut, $fanIn)
            ->multiplyScalar($scale);
    }
}
