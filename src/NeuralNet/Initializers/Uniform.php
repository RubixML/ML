<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
        return Matrix::uniform($fanOut, $fanIn)
            ->multiply($this->beta);
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
