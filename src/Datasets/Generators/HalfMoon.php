<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use InvalidArgumentException;

/**
 * Half Moon
 *
 * Generate a dataset consisting of 2-d samples that form a half moon shape.
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
     * @var \Rubix\Tensor\Vector
     */
    protected $center;

    /**
     * The scaling factor of the half moon.
     *
     * @var float
     */
    protected $scale;

    /**
     * The rotation on the half moon in degrees.
     *
     * @var float
     */
    protected $rotation;

    /**
     * The amount of gaussian noise to add to the points as a percentage.
     *
     * @var float
     */
    protected $noise;

    /**
     * @param  float  $x
     * @param  float  $y
     * @param  float  $scale
     * @param  float  $rotation
     * @param  float  $noise
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $x = 0., float $y = 0., float $scale = 1.0, float $rotation = 90.0,
                                float $noise = 0.1)
    {
        if ($scale < 0) {
            throw new InvalidArgumentException('Scaling factor must be greater'
                . ' than 0.');
        }

        if ($rotation < 0 or $rotation > 360) {
            throw new InvalidArgumentException('Rotation must be between 0 and'
                . ' 360 degrees.');
        }

        if ($noise <= 0.or $noise > 1.) {
            throw new InvalidArgumentException('Noise factor must be great than'
                . ' 0 and less than or equal to 1.');
        }

        $this->center = Vector::quick([$x, $y]);
        $this->scale = $scale;
        $this->rotation = $rotation;
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
     * @param  int  $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n = 100) : Dataset
    {
        $r = Vector::rand($n)->multiply(180)
            ->add($this->rotation)
            ->radians();

        $noise = Matrix::gaussian($n, 2)
            ->multiply($this->noise);
            
        $samples = Matrix::fromVectors([$r->cos(), $r->sin()])
            ->transpose()
            ->add($noise)
            ->multiply($this->scale)
            ->add($this->center)
            ->asArray();

        return Unlabeled::quick($samples);
    }

}
