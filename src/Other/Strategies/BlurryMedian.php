<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Gaussian;
use InvalidArgumentException;
use RuntimeException;

/**
 * Blurry Median
 *
 * A robust strategy that uses the median and median absolute deviation (MAD)
 * of the fitted data to make guesses.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BlurryMedian implements Continuous
{
    /**
     * The amount of gaussian noise to add to the guess.
     *
     * @var float
     */
    protected $blur;

    /**
     * The median of the fitted values.
     *
     * @var float|null
     */
    protected $median;

    /**
     * The median absolute deviation of the fitted values.
     *
     * @var float|null
     */
    protected $mad;

    /**
     * @param  float  $blur
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $blur = 0.3)
    {
        if ($blur < 0.or $blur > 1.) {
            throw new InvalidArgumentException('Blur factor must be between 0'
                . ' and 1.');
        }

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
            throw new InvalidArgumentException('Strategy needs to be fit with'
                . ' at least one value.');
        }

        list($this->median, $this->mad) = Stats::medMad($values);
    }

    /**
     * Make a continuous guess.
     *
     * @throws \RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if (is_null($this->median) or is_null($this->mad)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->median + $this->blur * Gaussian::rand() * $this->mad;
    }
}
