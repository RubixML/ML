<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use MathPHP\Probability\Distribution\Continuous\Normal;
use InvalidArgumentException;
use RuntimeException;

/**
 * Blurry Mean
 *
 * This continuous Strategy that adds a blur factor to the mean of a set of
 * values producing a random guess around the mean.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BlurryMean implements Continuous
{
    /**
     * The amount of gaussian noise as a factor of one standard deviation to add
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
        if ($blur < 0.) {
            throw new InvalidArgumentException('Blurr factor must be between 0'
                . ' and 1.');
        }

        $this->blur = $blur;
    }

    /**
     * Fit the strategy to the given values.
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

        $mean = Stats::mean($values);

        $this->distribution = new Normal($mean, $this->blur * $mean);
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @throws \RuntimeException
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
