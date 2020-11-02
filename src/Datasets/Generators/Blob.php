<?php

namespace Rubix\ML\Datasets\Generators;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Blob
 *
 * A normally distributed n-dimensional blob of samples centered at a given
 * mean vector. The standard deviation can be set for the whole blob or for each
 * feature column independently. When a global standard deviation is used, the
 * resulting blob will be isotropic and will converge asymptotically to a sphere.
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
    protected $stdDev;

    /**
     * @param (int|float)[] $center
     * @param int|float|(int|float)[] $stdDev
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $center = [0, 0], $stdDev = 1.0)
    {
        if (empty($center)) {
            throw new InvalidArgumentException('Cannot generate samples'
                . ' with dimensionality less than 1.');
        }

        if (is_array($stdDev)) {
            if (count($center) !== count($stdDev)) {
                throw new InvalidArgumentException('Number of center'
                    . ' coordinates and standard deviations must be equal.');
            }

            foreach ($stdDev as $value) {
                if ($value < 0) {
                    throw new InvalidArgumentException('Standard deviation'
                        . " must be greater than 0, $value given.");
                }
            }

            $stdDev = Vector::quick($stdDev);
        } else {
            if ($stdDev < 0) {
                throw new InvalidArgumentException('Standard deviation'
                    . " must be greater than 0, $stdDev given.");
            }
        }

        $this->center = Vector::quick($center);
        $this->stdDev = $stdDev;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @internal
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
     * @return \Rubix\ML\Datasets\Unlabeled
     */
    public function generate(int $n) : Unlabeled
    {
        $d = $this->dimensions();

        $samples = Matrix::gaussian($n, $d)
            ->multiply($this->stdDev)
            ->add($this->center)
            ->asArray();

        return Unlabeled::quick($samples);
    }
}
