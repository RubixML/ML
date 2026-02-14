<?php

namespace Rubix\ML\Datasets\Generators\SwissRoll;

use NDArray;
use NumPower;
use Rubix\ML\Datasets\Generators\Generator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function cos;
use function sin;
use function log;
use function sqrt;
use function mt_rand;

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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class SwissRoll implements Generator
{
    /**
     * The center vector of the swiss roll.
     *
     * @var list<float>
     */
    protected array $center;

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
     * @throws InvalidArgumentException
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

        $this->center = [$x, $y, $z];
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
     * @return Labeled
     */
    public function generate(int $n) : Labeled
    {
        $range = M_PI + HALF_PI;

        $t = [];
        $y = [];
        $coords = [];

        for ($i = 0; $i < $n; ++$i) {
            $u = mt_rand() / mt_getrandmax();
            $ti = (($u * 2.0) + 1.0) * $range;
            $t[] = $ti;

            $uy = mt_rand() / mt_getrandmax();
            $y[] = $uy * $this->depth;

            $coords[] = [
                $ti * cos($ti),
                $y[$i],
                $ti * sin($ti),
            ];
        }

        $noise = [];

        if ($this->noise > 0.0) {
            for ($i = 0; $i < $n; ++$i) {
                $row = [];

                for ($j = 0; $j < 3; ++$j) {
                    $u1 = mt_rand() / mt_getrandmax();
                    $u2 = mt_rand() / mt_getrandmax();
                    $u1 = $u1 > 0.0 ? $u1 : 1e-12;

                    $z0 = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);

                    $row[] = $z0 * $this->noise;
                }

                $noise[] = $row;
            }
        } else {
            for ($i = 0; $i < $n; ++$i) {
                $noise[] = [0.0, 0.0, 0.0];
            }
        }

        $center = [];

        for ($i = 0; $i < $n; ++$i) {
            $center[] = $this->center;
        }

        $coords = NumPower::array($coords);
        $noise = NumPower::array($noise);
        $center = NumPower::array($center);

        $samples = NumPower::add(
            NumPower::add(
                NumPower::multiply($coords, $this->scale),
                $center
            ),
            $noise
        );

        return Labeled::quick($samples->toArray(), $t);
    }
}
