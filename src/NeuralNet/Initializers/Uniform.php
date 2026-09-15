<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Uniform
 *
 * Generates a random uniform distribution centered at 0 and bounded at
 * both ends by the parameter beta.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Uniform implements Initializer
{
    /**
     * The upper and lower bound of the distribution.
     *
     * @var float
     */
    protected float $beta;

    /**
     * @param float $beta
     * @throws InvalidArgumentException
     */
    public function __construct(float $beta = 0.5)
    {
        if ($beta <= 0.0) {
            throw new InvalidArgumentException('Beta cannot be less than'
                . " or equal to 0, $beta given.");
        }

        $this->beta = $beta;
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

                $tensor = ColumnVector::uniform($n)->multiply($this->beta);

                break;

            case 2:
                [$m, $n] = $shape;

                $tensor = Matrix::uniform($m, $n)->multiply($this->beta);

                break;

            default:
                throw new InvalidArgumentException('Invalid shape for Uniform initializer.');
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
        return "Uniform (beta: {$this->beta})";
    }
}
