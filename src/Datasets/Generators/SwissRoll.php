<?php

namespace Rubix\ML\Datasets\Generators;

use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Swiss Roll
 *
 * Generate a 3-dimensional swiss roll dataset with continuous valued labels.
 * The labels are the inputs to the swiss roll transformation and are suitable
 * for non-linear regression problems.
 *
 * References:
 * [1] S. Marsland. (2009). Machine Learning: An Algorithmic Perspective,
 * Chapter 10.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SwissRoll implements Generator
{
    /**
     * The center vector of the swiss roll.
     *
     * @var \Rubix\Tensor\Vector
     */
    protected $center;

    /**
     * The scaling factor of the swiss roll.
     *
     * @var float
     */
    protected $scale;

    /**
     * The depth of the swiss roll i.e the scale of the y dimension.
     *
     * @var float
     */
    protected $depth;

    /**
     * The standard deviation of the gaussian noise.
     *
     * @var float
     */
    protected $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $z
     * @param float $scale
     * @param float $depth
     * @param float $noise
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $x = 0.0,
        float $y = 0.0,
        float $z = 0.0,
        float $scale = 1.,
                                float $depth = 21.,
        float $noise = 0.3
    ) {
        if ($scale < 0.) {
            throw new InvalidArgumentException('Scaling factor must be greater'
                . " than 0, $scale given.");
        }

        if ($depth < 0) {
            throw new InvalidArgumentException('Depth cannot be less than 0'
                . " $depth given.");
        }

        if ($noise < 0.) {
            throw new InvalidArgumentException('Noise factor cannot be less'
                . " than 0, $noise given.");
        }

        $this->center = Vector::quick([$x, $y, $z]);
        $this->scale = $scale;
        $this->depth = $depth;
        $this->noise = $noise;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return 3;
    }

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n) : Dataset
    {
        $t = Vector::rand($n)->multiply(2)->add(1)
            ->multiply(1.5 * M_PI);

        $x = $t->multiply($t->cos());
        $y = Vector::rand($n)->multiply($this->depth);
        $z = $t->multiply($t->sin());

        $noise = Matrix::gaussian($n, 3)
            ->multiply($this->noise);
            
        $samples = Matrix::stack([$x, $y, $z])
            ->transpose()
            ->add($noise)
            ->multiply($this->scale)
            ->add($this->center)
            ->asArray();

        $labels = $t->asArray();

        return Labeled::quick($samples, $labels);
    }
}
