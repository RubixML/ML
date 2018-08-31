<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use MathPHP\Probability\Distribution\Continuous\Normal;
use InvalidArgumentException;
use RuntimeException;

/**
 * Blurry Median
 *
 * Adds random Gaussian noise to the median of a set of values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BlurryMedian implements Continuous
{
    /**
     * The amount of gaussian noise as a factor of one interquartile range to add
     * to the guess.
     *
     * @var float
     */
    protected $blur;

    /**
     * The probability distribution to sample from.
     *
     * @var \MathPHP\Probability\Distribution\Continuous\Normal|null
     */
    protected $distribution;

    /**
     * @param  float  $blur
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $blur = 0.1)
    {
        if ($blur < 0.or $blur > 1.) {
            throw new InvalidArgumentException('Blur factor must be between 0'
                . ' and 1.');
        }

        $this->blur = $blur;
    }

    /**
     * Fit the imputer to the feature column of the training data.
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

        $median = Stats::median($values);

        $this->distribution = new Normal($median, $this->blur * $median);
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @return mixed
     */
    public function guess()
    {
        if (is_null($this->distribution)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->distribution->rand();
    }
}
