<?php

namespace Rubix\ML\Datasets\Generators;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use InvalidArgumentException;

/**
 * Blob
 *
 * A normally distributed n-dimensional blob of samples centered at a given
 * mean vector. The standard deviation can be set for the whole blob or for each
 * feature column independently. When a global standard deviation is used, the
 * resulting blob will be isotropic and will converge asypmtotically to a sphere.
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
     * @var \Tensor\Vector
     */
    protected $center;

    /**
     * The standard deviation of the blob.
     *
     * @var \Tensor\Vector|int|float
     */
    protected $stddev;

    /**
     * @param array $center
     * @param mixed $stddev
     * @throws \InvalidArgumentException
     */
    public function __construct(array $center = [0, 0], $stddev = 1.)
    {
        if (empty($center)) {
            throw new InvalidArgumentException('Cannot generate data of less'
                . ' than 1 dimension.');
        }

        foreach ($center as $value) {
            if (!is_int($value) and !is_float($value)) {
                throw new InvalidArgumentException('Center coordinate must be'
                    . ' an integer or floating point number, '
                    . gettype($value) . ' given');
            }
        }

        if (is_array($stddev)) {
            if (count($center) !== count($stddev)) {
                throw new InvalidArgumentException('The number of center'
                    . ' coordinates and standard deviations must equal.');
            }

            foreach ($stddev as $value) {
                if (!is_int($value) and !is_float($value)) {
                    throw new InvalidArgumentException('Standard deviation must be'
                        . ' an integer or float, ' . gettype($value) . ' given.');
                }
    
                if ($value <= 0) {
                    throw new InvalidArgumentException('Standard deviation must be'
                        . " greater than 0, $value given.");
                }
            }

            $stddev = Vector::quick($stddev);
        } else {
            if (!is_int($stddev) and !is_float($stddev)) {
                throw new InvalidArgumentException('Standard deviation must'
                    . ' be an integer or floating point number, '
                    . gettype($stddev) . ' given');
            }
        }

        $this->center = Vector::quick($center);
        $this->stddev = $stddev;
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
     * @param int $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n) : Dataset
    {
        $d = $this->dimensions();
        
        $samples = Matrix::gaussian($n, $d)
            ->multiply($this->stddev)
            ->add($this->center)
            ->asArray();

        return Unlabeled::quick($samples);
    }
}
