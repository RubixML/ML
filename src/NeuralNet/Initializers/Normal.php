<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
        return Matrix::gaussian($fanOut, $fanIn)
            ->multiply($this->stdDev);
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
