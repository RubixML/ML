<?php

namespace Rubix\ML\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Functions\Gaussian;
use MathPHP\Probability\Distribution\Continuous\Normal;
use MathPHP\Probability\Distribution\Continuous\Uniform;
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
     * The x and y coordinates of the center of the half moon.
     *
     * @var array
     */
    protected $center;

    /**
     * The scaling factor of the moon.
     *
     * @var float
     */
    protected $scale;

    /**
     * The uniform probability distribution to sample from.
     *
     * @var \MathPHP\Probability\Distribution\Continuous\Uniform
     */
    protected $uniform;

    /**
     * The normal probability distribution to sample from.
     *
     * @var \MathPHP\Probability\Distribution\Continuous\Normal
     */
    protected $gaussian;

    /**
     * @param  array  $center
     * @param  float  $scale
     * @param  float  $rotate
     * @param  float  $noise
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $center = [0, 0], float $scale = 1.0, float $rotate = 90.0,
                                float $noise = 0.1)
    {
        if (count($center) !== 2) {
            throw new InvalidArgumentException('This generator only works in 2'
                . ' dimensions.');
        }

        if ($scale < 0) {
            throw new InvalidArgumentException('Scaling factor must be greater'
                . ' than 0.');
        }

        if ($rotate < 0 or $rotate > 360) {
            throw new InvalidArgumentException('Rotation must be between 0 and'
                . ' 360 degrees.');
        }

        if ($noise <= 0.or $noise > 1.) {
            throw new InvalidArgumentException('Noise factor must be great than'
                . ' 0 and less than or equal to 1.');
        }

        $this->center = $center;
        $this->scale = $scale;
        $this->uniform = new Uniform(deg2rad($rotate), deg2rad($rotate + 180));
        $this->gaussian = new Normal(0, $scale * $noise);
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
        $samples = [];

        for ($i = 0; $i < $n; $i++) {
            $random = $this->uniform->rand();

            $samples[$i][0] = $this->scale * cos($random)
                + $this->center[0]
                + $this->gaussian->rand();
            $samples[$i][1] = $this->scale * sin($random)
                + $this->center[1]
                + $this->gaussian->rand();
        }

        return new Unlabeled($samples, false);
    }

}
