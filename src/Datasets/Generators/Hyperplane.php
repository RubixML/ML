<?php

namespace Rubix\ML\Datasets\Generators;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Hyperplane
 *
 * Generates a labeled dataset whose samples form a hyperplane in n-dimensional vector
 * space and whose labels are continuous values drawn from a uniform random distribution
 * between -1 and 1. When the number of coefficients is either 1, 2 or 3, the samples
 * form points, lines, and planes respectively. Due to its linearity, Hyperplane is
 * especially useful for testing linear regression models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Hyperplane implements Generator
{
    /**
     * The n coefficients of the hyperplane where n is the dimensionality.
     *
     * @var \Tensor\Vector
     */
    protected \Tensor\Vector $coefficients;

    /**
     * The y intercept term.
     *
     * @var float
     */
    protected float $intercept;

    /**
     * The factor of gaussian noise to add to the data points.
     *
     * @var float
     */
    protected float $noise;

    /**
     * @param (int|float)[] $coefficients
     * @param float $intercept
     * @param float $noise
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        array $coefficients = [1, -1],
        float $intercept = 0.0,
        float $noise = 0.1
    ) {
        if (empty($coefficients)) {
            throw new InvalidArgumentException('Cannot generate samples'
                . ' with dimensionality less than 1.');
        }

        if ($noise < 0.0) {
            throw new InvalidArgumentException('Noise must be'
                . " greater than 0, $noise given.");
        }

        $this->coefficients = Vector::quick($coefficients);
        $this->intercept = $intercept;
        $this->noise = $noise;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @internal
     *
     * @return int<0,max>
     */
    public function dimensions() : int
    {
        return $this->coefficients->n();
    }

    /**
     * Generate n data points.
     *
     * @param int<0,max> $n
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function generate(int $n) : Labeled
    {
        $d = $this->dimensions();

        $y = Vector::uniform($n);

        $noise = Matrix::gaussian($n, $d)
            ->multiply($this->noise);

        $samples = $y->add($this->intercept)
            ->asColumnMatrix()
            ->repeat(0, $d - 1)
            ->multiply($this->coefficients)
            ->add($noise)
            ->asArray();

        $labels = $y->asArray();

        return Labeled::quick($samples, $labels);
    }
}
