<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Descriptive;
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
     * The amount of gaussian noise by ratio of the interquartile range to add
     * to the guess.
     *
     * @var float
     */
    protected $blur;

    /**
     * The precomputed median of the fitted data.
     *
     * @var float|null
     */
    protected $median;

    /**
     * The precomputed interquartile range of the fitted data.
     *
     * @var float|null
     */
    protected $iqr;

    /**
     * @param  float  $blur
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $blur = 0.05)
    {
        if ($blur < 0.0 or $blur > 1.0) {
            throw new InvalidArgumentException('Blur factor must be between 0'
                . ' and 1.');
        }

        $this->blur = $blur;
    }

    /**
     * Return the range of possible guesses for this strategy in a tuple.
     *
     * @return array
     */
    public function range() : array
    {
        $r = $this->blur * $this->iqr;

        return [$this->median - $r, $this->median + $r];
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

        $quartiles = Descriptive::quartiles($values);

        $this->median = $quartiles['Q2'];
        $this->iqr = $quartiles['IQR'];
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @return mixed
     */
    public function guess()
    {
        if (is_null($this->median) or is_null($this->iqr)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->median + $this->blur
            * $this->generateGaussianValue()
            * $this->iqr;
    }

    /**
     * Generate a float value between -1 and 1.
     *
     * @return float
     */
    protected function generateGaussianValue() : float
    {
        return rand((int) (-1 * 1e8), (int) (1 * 1e8)) / 1e8;
    }
}
