<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Constant
 *
 * Initialize the parameter to a user specified constant value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Initializer
{
    /**
     * The value to initialize the parameter to.
     *
     * @var float
     */
    protected float $value;

    /**
     * @param float $value
     * @throws InvalidArgumentException
     */
    public function __construct(float $value = 0.0)
    {
        if (is_nan($value)) {
            throw new InvalidArgumentException('Cannot initialize'
                . ' weight values to NaN.');
        }

        $this->value = $value;
    }

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

                $tensor = ColumnVector::fill($this->value, $n);

                break;

            case 2:
                [$m, $n] = $shape;

                $tensor = Matrix::fill($this->value, $m, $n);

                break;

            default:
                throw new InvalidArgumentException('Invalid shape for Constant initializer.');
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
        return "Constant (value: {$this->value})";
    }
}
