<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Gaussian;
use InvalidArgumentException;
use RuntimeException;

/**
 * Blurry Percentile
 *
 * A strategy that guesses within the domain of the p-th percentile of the
 * fitted data plus some gaussian noise.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BlurryPercentile implements Continuous
{
    const TWO_PI = 2. * M_PI;

    /**
     * The index of the percentile to predict where 50 is the median.
     *
     * @var float
     */
    protected $p;

    /**
     * The amount of gaussian noise to add to the guess as a factor of the
     * median absolute deviation of the fitted data.
     *
     * @var float
     */
    protected $blur;

    /**
     * The pth percentile of the fitted values.
     *
     * @var float|null
     */
    protected $percentile;

    /**
     * The median absolute deviation of the fitted values.
     *
     * @var float|null
     */
    protected $mad;

    /**
     * @param  float  $p
     * @param  float  $blur
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $p = 50.0, float $blur = 0.1)
    {
        if ($p < 0. or $p > 100.) {
            throw new InvalidArgumentException('Percentile must be between 0'
                . " and 100, $p given.");
        }

        if ($blur < 0. or $blur > 1.) {
            throw new InvalidArgumentException('Blur factor must be between 0'
                . " and 1, $blur given.");
        }

        $this->p = $p;
        $this->blur = $blur;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param  array $values
     * @return void
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fit with'
                . ' at least 1 value.');
        }

        $this->percentile = Stats::percentile($values, $this->p);
        $this->mad = Stats::mad($values) ?: self::EPSILON;
    }

    /**
     * Make a guess.
     *
     * @throws \RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if (is_null($this->percentile) or is_null($this->mad)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->percentile + $this->blur * $this->gaussian() * $this->mad;
    }

    /**
     * Generate a random number from a Gaussian distribution with 0 mean and
     * standard deviation of 1 i.e a number between -1 and 1.
     *
     * @return float
     */
    public static function gaussian() : float
    {
        $r1 = rand(0, PHP_INT_MAX) / PHP_INT_MAX;
        $r2 = rand(0, PHP_INT_MAX) / PHP_INT_MAX;

        return sqrt(-2. * log($r1)) * cos(self::TWO_PI * $r2);
    }
}
