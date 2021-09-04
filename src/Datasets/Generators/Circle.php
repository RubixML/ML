<?php

namespace Rubix\ML\Datasets\Generators;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function Rubix\ML\array_transpose;

use const Rubix\ML\TWO_PI;

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
    /**
     * The center vector of the circle.
     *
     * @var \Tensor\Vector
     */
    protected \Tensor\Vector $center;

    /**
     * The scaling factor of the circle.
     *
     * @var float
     */
    protected float $scale;

    /**
     * The factor of gaussian noise to add to the data points.
     *
     * @var float
     */
    protected float $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $scale
     * @param float $noise
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        float $x = 0.0,
        float $y = 0.0,
        float $scale = 1.0,
        float $noise = 0.1
    ) {
        if ($scale < 0.0) {
            throw new InvalidArgumentException('Scale must be'
                . " greater than 0, $scale given.");
        }

        if ($noise < 0.0) {
            throw new InvalidArgumentException('Noise must be'
                . " greater than 0, $noise given.");
        }

        $this->center = Vector::quick([$x, $y]);
        $this->scale = $scale;
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
        return 2;
    }

    /**
     * Generate n data points.
     *
     * @param int<0,max> $n
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function generate(int $n) : Labeled
    {
        $r = Vector::rand($n)->multiply(TWO_PI);

        $x = $r->cos()->asArray();
        $y = $r->sin()->asArray();

        $coordinates = array_transpose([$x, $y]);

        $noise = Matrix::gaussian($n, 2)
            ->multiply($this->noise);

        $samples = Matrix::quick($coordinates)
            ->multiply($this->scale)
            ->add($this->center)
            ->add($noise)
            ->asArray();

        $labels = $r->rad2deg()->asArray();

        return Labeled::quick($samples, $labels);
    }
}
