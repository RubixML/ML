<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

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
     * The minimum and maximum bound on the random distribution.
     *
     * @var float
     */
    protected $beta;

    /**
     * @param  float  $beta
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $beta = 0.5)
    {
        if ($beta <= 0.) {
            throw new InvalidArgumentException('Beta cannot be less than'
                . " or equal to 0, $beta given.");
        }

        $this->beta = $beta;
    }

    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @param  int  $fanIn
     * @param  int  $fanOut
     * @return \Rubix\Tensor\Matrix
     */
    public function init(int $fanIn, int $fanOut) : Matrix
    {
        return Matrix::uniform($fanOut, $fanIn)
            ->multiply($this->beta);
    }
}
