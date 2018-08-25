<?php

namespace Rubix\ML\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use MathPHP\Probability\Distribution\Continuous\Normal;
use InvalidArgumentException;

/**
 * Blob
 *
 * A normally distributed n-dimensional blob of samples centered at a given
 * mean vector. The standard deviation can be set for the whole blob or for each
 * feature column independently. When a global value is used, the resulting blob
 * will be isotropic.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Blob implements Generator
{
    /**
     * The probability distributions of each feature column.
     *
     * @var array
     */
    protected $distributions = [
        //
    ];

    /**
     * @param  array  $center
     * @param  mixed  $stddev
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $center = [0.0, 0.0], $stddev = 1.)
    {
        if (count($center) === 0) {
            throw new InvalidArgumentException('Cannot generate data of less'
                . ' than 1 dimension.');
        }

        if (!is_array($stddev)) {
            $stddev = array_fill(0, count($center), (float) $stddev);
        }

        if (count($center) !== count($stddev)) {
            throw new InvalidArgumentException('The number of center'
                . ' coordinates and standard deviations must be equal.');
        }

        foreach ($center as $column => $mean) {
            if (!is_int($mean) and !is_float($mean)) {
                throw new InvalidArgumentException('Center coordinate must be'
                    . ' a numeric type.');
            }

            if (!is_int($stddev[$column]) and !is_float($stddev[$column])) {
                throw new InvalidArgumentException('Standard deviation must be'
                    . ' a numeric type.');
            }

            if ($stddev[$column] <= 0) {
                throw new InvalidArgumentException('Standard deviation must be'
                 . ' 0 or above.');
            }

            $this->distributions[] = new Normal($mean, $stddev[$column]);
        }
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return count($this->distributions);
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
            foreach ($this->distributions as $column => $distribution) {
                $samples[$i][$column] = $distribution->rand();
            }
        }

        return new Unlabeled($samples, false);
    }
}
