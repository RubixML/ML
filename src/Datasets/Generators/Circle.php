<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\Gaussian;
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
    const TWO_PI = 2. * M_PI;
    const PHI = 1000000000;

    /**
     * The x coordinate of the center of the circle.
     *
     * @var float
     */
    protected $x;

    /**
     * The y coordinate of the center of the circle.
     *
     * @var float
     */
    protected $y;

    /**
     * The scaling factor of the circle.
     *
     * @var float
     */
    protected $scale;

    /**
     * The standard deviation of the generated data points.
     *
     * @var float
     */
    protected $stddev;

    /**
     * @param  float  $x
     * @param  float  $y
     * @param  float  $scale
     * @param  float  $noise
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $x = 0., float $y = 0., float $scale = 1.0, float $noise = 0.1)
    {
        if ($scale < 0.) {
            throw new InvalidArgumentException('Scaling factor must be greater'
                . ' than 0.');
        }

        if ($noise <= 0.or $noise > 1.) {
            throw new InvalidArgumentException('Noise factor must be great than'
                . ' 0 and less than or equal to 1.');
        }

        $this->x = $x;
        $this->y = $y;
        $this->scale = $scale;
        $this->stddev = $noise * $scale;
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
        $end = (int) round(self::TWO_PI * self::PHI);

        $samples = [];

        for ($i = 0; $i < $n; $i++) {
            $r = rand(0, $end) / self::PHI;

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
