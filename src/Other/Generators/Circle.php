<?php

namespace Rubix\ML\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use InvalidArgumentException;

class Circle implements Generator
{
    /**
     * The x and y coordinates of the center of the circle.
     *
     * @var array
     */
    protected $center;

    /**
     * The center mean of the blob i.e. the mean vector.
     *
     * @var float
     */
    protected $scale;

    /**
     * The amount of noise to add to each feature column i.e the standard
     * deviation of the gaussian noise added to the mean.
     *
     * @var float
     */
    protected $noise;

    /**
     * @param  array  $center
     * @param  float  $scale
     * @param  float  $noise
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $center = [0, 0], float $scale = 1.0, float $noise = 1.0)
    {
        if (count($center) !== 2) {
            throw new InvalidArgumentException('This generator only works in 2'
                . ' dimensions.');
        }

        if ($scale < 0) {
            throw new InvalidArgumentException('Scaling factor must be greater'
                . ' than 0.');
        }

        if ($noise < 0) {
            throw new InvalidArgumentException('Noise factor must be'
                . ' positive.');
        }

        $this->center = $center;
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
     * @param  int  $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n = 100) : Dataset
    {
        $samples = [];

        $spacing = $n / (2.0 * M_PI);

        for ($i = 0; $i < $n; $i++) {
            $samples[$i][0] = ($this->scale * cos($i * $spacing))
                + $this->center[0];
            $samples[$i][1] = ($this->scale * sin($i * $spacing))
                + $this->center[1];
        }

        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                $feature += $this->generateRandomGaussian() * $this->noise;
            }
        }

        return new Unlabeled($samples);
    }

    /**
     * Generate a float value between -1 and 1.
     *
     * @return float
     */
    protected function generateRandomGaussian() : float
    {
        return rand((int) (-1 * 1e8), (int) (1 * 1e8)) / 1e8;
    }
}
