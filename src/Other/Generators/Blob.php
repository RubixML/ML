<?php

namespace Rubix\ML\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use InvalidArgumentException;

class Blob implements Generator
{
    /**
     * The center mean of the blob i.e. the mean vector.
     *
     * @var array
     */
    protected $center = [
        //
    ];

    /**
     * The standard deviations from the mean of the blob.
     *
     * @var array
     */
    protected $stddev = [
        //
    ];

    /**
     * @param  array  $center
     * @param  mixed  $stddev
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $center = [0.0, 0.0], $stddev = 1.0)
    {
        if (count($center) === 0) {
            throw new InvalidArgumentException('Cannot generate data of less'
                . ' than 1 dimension.');
        }

        if (!is_array($stddev)) {
            $stddev = array_fill(0, count($center), $stddev);
        }

        if (count($center) !== count($stddev)) {
            throw new InvalidArgumentException('The number of center'
                . ' coordinates and standard deviations must be equal.');
        }

        foreach ($center as $column => $coordinate) {
            if (!is_int($coordinate) and !is_float($coordinate)) {
                throw new InvalidArgumentException('Center coordinate must be'
                    . ' a numeric type.');
            }

            if (!is_int($stddev[$column]) and !is_float($stddev[$column])) {
                throw new InvalidArgumentException('Center coordinate must be'
                    . ' a numeric type.');
            }

            if ($stddev[$column] < 0) {
                throw new InvalidArgumentException('Standard deviation must be'
                    . ' positive.');
            }
        }

        $this->center = $center;
        $this->stddev = $stddev;
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
            foreach ($this->center as $column => $mean) {
                $samples[$i][$column] = $mean
                    + $this->generateRandomGaussian()
                    * $this->stddev[$column];
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
