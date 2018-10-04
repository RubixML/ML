<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\Gaussian;
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
    const PHI = 1000000000;

    /**
     * The x coordinate of the center of the half moon.
     *
     * @var float
     */
    protected $x;

    /**
     * The y coordinate of the center of the half moon.
     *
     * @var float
     */
    protected $y;

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
     * The standard deviation of the gaussian noise added to the data.
     *
     * @var float
     */
    protected $stddev;

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

        $this->x = $x;
        $this->y = $y;
        $this->scale = $scale;
        $this->rotation = $rotation;
        $this->stddev = $scale * $noise;
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
        $start = (int) round(deg2rad($this->rotation) * self::PHI);
        $end = (int) round(deg2rad($this->rotation + 180) * self::PHI);

        $samples = [];

        for ($i = 0; $i < $n; $i++) {
            $r = rand($start, $end) / self::PHI;

            $samples[$i][] = $this->scale * cos($r)
                + $this->x
                + Gaussian::rand() * $this->stddev;

            $samples[$i][] = $this->scale * sin($r)
                + $this->y
                + Gaussian::rand() * $this->stddev;
        }

        return new Unlabeled($samples, false);
    }

}
