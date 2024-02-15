<?php

namespace Rubix\ML\Datasets\Generators;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function Rubix\ML\array_transpose;

/**
 * Half Moon
 *
 * Generates a dataset consisting of 2-d samples that form the shape of a half moon when
 * plotted on a scatter plot chart.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HalfMoon implements Generator
{
    /**
     * The center vector of the circle.
     *
     * @var Vector
     */
    protected Vector $center;

    /**
     * The scaling factor of the half moon.
     *
     * @var float
     */
    protected float $scale;

    /**
     * The rotation on the half moon in degrees.
     *
     * @var float
     */
    protected float $rotation;

    /**
     * The factor of Gaussian noise to add to the data points.
     *
     * @var float
     */
    protected float $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $scale
     * @param float $rotation
     * @param float $noise
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $x = 0.0,
        float $y = 0.0,
        float $scale = 1.0,
        float $rotation = 90.0,
        float $noise = 0.1
    ) {
        if ($scale < 0.0) {
            throw new InvalidArgumentException('Scale must be'
                . " greater than 0, $scale given.");
        }

        if ($rotation < 0.0 or $rotation > 360.0) {
            throw new InvalidArgumentException('Rotation must be'
                . " between 0 and 360, $rotation given.");
        }

        if ($noise < 0.0) {
            throw new InvalidArgumentException('Noise must be'
                . " greater than 0, $noise given.");
        }

        $this->center = Vector::quick([$x, $y]);
        $this->scale = $scale;
        $this->rotation = $rotation;
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
     * @return Labeled
     */
    public function generate(int $n) : Labeled
    {
        $r = Vector::rand($n)->multiply(M_PI)
            ->add(deg2rad($this->rotation));

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
