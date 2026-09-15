<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function sqrt;

/**
 * Xavier 1
 *
 * The Xavier 1 initializer draws from a uniform distribution [-limit, limit]
 * where *limit* is equal to sqrt(6 / (fanIn + fanOut)). This initializer is
 * best suited for layers that feed into an activation layer that outputs a
 * value between 0 and 1 such as Sigmoid.
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

                $tensor = ColumnVector::uniform($n)->multiply(sqrt(6.0 / ($n + 1)));

                break;

            case 2:
                [$m, $n] = $shape;

                $tensor = Matrix::uniform($m, $n)->multiply(sqrt(6.0 / ($m + $n)));

                break;

            default:
                throw new InvalidArgumentException('Invalid shape for Xavier 1 initializer.');
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
        return 'Xavier 1';
    }
}
