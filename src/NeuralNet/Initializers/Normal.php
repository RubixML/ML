<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Normal
 *
 * Generates a random weight matrix from a Gaussian distribution with user-specified standard
 * deviation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Normal implements Initializer
{
    /**
     * The standard deviation of the distribution to sample from.
     *
     * @var float
     */
    protected float $stdDev;

    /**
     * @param float $stdDev
     * @throws InvalidArgumentException
     */
    public function __construct(float $stdDev = 0.05)
    {
        if ($stdDev <= 0.0) {
            throw new InvalidArgumentException('Standard deviation must'
                . " be greater than 0, $stdDev given.");
        }

        $this->stdDev = $stdDev;
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

                $tensor = ColumnVector::gaussian($n)->multiply($this->stdDev);

                break;

            case 2:
                [$m, $n] = $shape;

                $tensor = Matrix::gaussian($m, $n)->multiply($this->stdDev);

                break;

            default:
                throw new InvalidArgumentException('Invalid shape for Normal initializer.');
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
        return "Normal (std_dev: {$this->stdDev})";
    }
}
