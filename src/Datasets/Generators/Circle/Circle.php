<?php

namespace Rubix\ML\Datasets\Generators\Circle;

use Rubix\ML\Datasets\Generators\Generator;
use NDArray;
use NumPower;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function array_map;

use const Rubix\ML\TWO_PI;

/**
 * Circle
 *
 * Create a circle made of sample data points in 2 dimensions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Circle implements Generator
{
    /**
     * The center vector of the circle.
     *
     * @var NDArray
     */
    protected NDArray $center;

    /**
     * The scaling factor of the circle.
     *
     * @var float
     */
    protected float $scale;

    /**
     * The factor of gaussian noise to add to the data points.
     *
     * @var float
     */
    protected float $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $scale
     * @param float $noise
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $x = 0.0,
        float $y = 0.0,
        float $scale = 1.0,
        float $noise = 0.1
    ) {
        if ($scale < 0.0) {
            throw new InvalidArgumentException('Scale must be'
                . " greater than 0, $scale given.");
        }

        if ($noise < 0.0) {
            throw new InvalidArgumentException('Noise must be'
                . " greater than 0, $noise given.");
        }

        $this->center = NumPower::array([$x, $y]);
        $this->scale = $scale;
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
        return 2;
    }

    /**
     * Generate n data points.
     *
     * @param int<0,max> $n
     * @return Labeled
     */
    public function generate(int $n) : Labeled
    {
        $r = NumPower::multiply(NumPower::uniform([$n]), TWO_PI);

        $angles = $r->toArray();

        $coordinates = array_map(
            static fn (float $angle) : array => [cos($angle), sin($angle)],
            $angles
        );

        $noise = NumPower::multiply(
            NumPower::normal([$n, 2]),
            $this->noise
        );

        $samples = NumPower::add(
            NumPower::add(
                NumPower::multiply(
                    NumPower::array($coordinates),
                    $this->scale
                ),
                $this->center
            ),
            $noise
        )->toArray();

        // Convert radians to degrees
        $labels = NumPower::multiply($r, 180.0 / M_PI)->toArray();

        return Labeled::quick($samples, $labels);
    }
}
