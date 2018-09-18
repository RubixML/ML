<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Blurry Mean
 *
 * This strategy adds a blur factor to the mean of a set of values producing a
 * random guess centered around the mean. The amount of blur is determined as
 * the blur factor times the standard deviation of the fitted data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BlurryMean implements Continuous
{
    const TWO_PI = 2. * M_PI;

    /**
     * The amount of gaussian noise to add to the guess.
     *
     * @var float
     */
    protected $blur;

    /**
     * The mean of the fitted values.
     *
     * @var float|null
     */
    protected $mean;

    /**
     * The standard deviation of the fitted values.
     *
     * @var float|null
     */
    protected $stddev;

    /**
     * @param  float  $blur
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $blur = 0.3)
    {
        if ($blur < 0.) {
            throw new InvalidArgumentException('Blur factor must be between 0'
                . ' and 1.');
        }

        $this->blur = $blur;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy needs to be fit with'
                . ' at least one value.');
        }

        list($mean, $variance) = Stats::meanVar($values);

        $this->mean = $mean;
        $this->stddev = sqrt($variance);
    }

    /**
     * Make a continuous guess.
     *
     * @throws \RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if (is_null($this->mean) or is_null($this->stddev)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->mean + $this->blur * $this->gaussian() * $this->stddev;
    }

    /**
     * Generate a random number from a gaussian distribution between -1 and 1.
     *
     * @return float
     */
    public function gaussian() : float
    {
        $r1 = rand(0, PHP_INT_MAX) / PHP_INT_MAX;
        $r2 = rand(0, PHP_INT_MAX) / PHP_INT_MAX;

        return sqrt(-2. * log($r1)) * cos(self::TWO_PI * $r2);
    }
}
