<?php

namespace Rubix\ML\Datasets\Generators;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function Rubix\ML\array_transpose;

use const Rubix\ML\HALF_PI;

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
     * @var \Tensor\Vector
     */
    protected \Tensor\Vector $center;

    /**
     * The scaling factor of the swiss roll.
     *
     * @var float
     */
    protected float $scale;

    /**
     * The depth of the swiss roll i.e the scale of the y dimension.
     *
     * @var float
     */
    protected float $depth;

    /**
     * The standard deviation of the gaussian noise.
     *
     * @var float
     */
    protected float $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $z
     * @param float $scale
     * @param float $depth
     * @param float $noise
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        float $x = 0.0,
        float $y = 0.0,
        float $z = 0.0,
        float $scale = 1.0,
        float $depth = 21.0,
        float $noise = 0.1
    ) {
        if ($scale < 0.0) {
            throw new InvalidArgumentException('Scale must be'
                . " greater than 0, $scale given.");
        }

        if ($depth < 0) {
            throw new InvalidArgumentException('Depth must be'
                . " greater than 0, $depth given.");
        }

        if ($noise < 0.0) {
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
     * @internal
     *
     * @return int<0,max>
     */
    public function dimensions() : int
    {
        return 3;
    }

    /**
     * Generate n data points.
     *
     * @param int<0,max> $n
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function generate(int $n) : Labeled
    {
        $t = Vector::rand($n)
            ->multiply(2)
            ->add(1)
            ->multiply(M_PI + HALF_PI);

        $x = $t->multiply($t->cos())->asArray();
        $y = Vector::rand($n)->multiply($this->depth)->asArray();
        $z = $t->multiply($t->sin())->asArray();

        $coordinates = array_transpose([$x, $y, $z]);

        $noise = Matrix::gaussian($n, 3)
            ->multiply($this->noise);

        $samples = Matrix::quick($coordinates)
            ->multiply($this->scale)
            ->add($this->center)
            ->add($noise)
            ->asArray();

        $labels = $t->asArray();

        return Labeled::quick($samples, $labels);
    }
}
