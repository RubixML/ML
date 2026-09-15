<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
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
     * Initialize a parameter tensor given its target shape.
     *
     * @internal
     *
     * @param int[] $shape
     * @return Parameter
     */
    public function initialize(array $shape) : Parameter
    {
        switch (count($shape)) {
            case 1:
                $n = $shape[0];

                $tensor = ColumnVector::uniform($n)->multiply(sqrt(3.0));

                break;

            case 2:
                [$m, $n] = $shape;

                $tensor = Matrix::uniform($m, $n)->multiply(sqrt(3.0 / $n));

                break;

            default:
                throw new InvalidArgumentException('Invalid shape for Le Cun initializer.');
        }

        return new Parameter($tensor);
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
