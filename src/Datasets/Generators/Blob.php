<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
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
     * The center vector of the blob.
     *
     * @var \Rubix\Tensor\Vector
     */
    protected $center;

    /**
     * The standard deviations of each dimension of the blob.
     *
     * @var \Rubix\Tensor\Vector
     */
    protected $stddev;

    /**
     * @param  array  $center
     * @param  mixed  $stddev
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $center = [0.0, 0.0], $stddev = 1.)
    {
        $d = count($center);

        if ($d < 1) {
            throw new InvalidArgumentException('Cannot generate data of less'
                . ' than 1 dimension.');
        }

        if (!is_array($stddev)) {
            $stddev = array_fill(0, $d, (float) $stddev);
        }

        if ($d !== count($stddev)) {
            throw new InvalidArgumentException("The number of center"
                . " coordinates and standard deviations must the same.");
        }

        foreach ($center as $column => $mean) {
            if (!is_int($mean) and !is_float($mean)) {
                throw new InvalidArgumentException('Center coordinate must be'
                    . ' an integer or float, ' . gettype($mean) . ' found.');
            }

            if (!is_int($stddev[$column]) and !is_float($stddev[$column])) {
                throw new InvalidArgumentException('Standard deviation must be'
                    . ' an integer or float, '. gettype($mean) . ' found.');
            }

            if ($stddev[$column] <= 0) {
                throw new InvalidArgumentException("Standard deviation must be"
                 . " greater than 0, $stddev[$column] given.");
            }
        }

        $this->center = Vector::quick($center);
        $this->stddev = Vector::quick($stddev);
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return $this->center->n();
    }

    /**
     * Generate n data points.
     *
     * @param  int  $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n = 100) : Dataset
    {
        $samples = Matrix::gaussian($n, $this->dimensions())
            ->multiply($this->stddev)
            ->add($this->center)
            ->asArray();

        return Unlabeled::quick($samples);
    }
}
