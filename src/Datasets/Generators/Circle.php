<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Circle
 *
 * Create a circle made of sample data points in 2 dimensions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Circle implements Generator
{
    protected const TWO_PI = 2. * M_PI;

    /**
     * The center vector of the circle.
     *
     * @var \Rubix\Tensor\Vector
     */
    protected $center;

    /**
     * The scaling factor of the circle.
     *
     * @var float
     */
    protected $scale;

    /**
     * The amount of gaussian noise to add to the points as a percentage.
     *
     * @var float
     */
    protected $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $scale
     * @param float $noise
     * @throws \InvalidArgumentException
     */
    public function __construct(float $x = 0.0, float $y = 0.0, float $scale = 1.0, float $noise = 0.1)
    {
        if ($scale < 0.) {
            throw new InvalidArgumentException('Scaling factor must be greater'
                . " than 0, $scale given.");
        }

        if ($noise < 0. or $noise > 1.) {
            throw new InvalidArgumentException('Noise factor must be between 0'
                . " and less 1, $noise given.");
        }

        $this->center = Vector::quick([$x, $y]);
        $this->scale = $scale;
        $this->noise = $noise;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return 2;
    }

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n) : Dataset
    {
        $r = Vector::rand($n)->multiply(self::TWO_PI);

        $x = $r->cos();
        $y = $r->sin();

        $noise = Matrix::gaussian($n, 2)
            ->multiply($this->noise);

        $samples = Matrix::stack([$x, $y])
            ->transpose()
            ->add($noise)
            ->multiply($this->scale)
            ->add($this->center)
            ->asArray();

        $labels = $r->degrees()->asArray();

        return Labeled::quick($samples, $labels);
    }
}
