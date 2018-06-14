<?php

namespace Rubix\ML\Transformers\Strategies;

use MathPHP\Statistics\Average;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class BlurryMean implements Continuous
{
    /**
     * The amount of gaussian noise by ratio of the variance to add to the mean.
     *
     * @var float
     */
    protected $blurr;

    /**
     * The precomputed mean of the fitted feature column.
     *
     * @var array
     */
    protected $mean;

    /**
     * The precomputed standard deviation of the fitted feature column.
     *
     * @var array
     */
    protected $stddev;

    /**
     * @param  float  $blurr
     * @return void
     */
    public function __construct(float $blurr = 0.1)
    {
        if ($blurr < 0.0 or $blurr > 1.0) {
            throw new InvalidArgumentException('Blurr factor must be between 0'
                . ' and 1.');
        }

        $this->blurr = $blurr;
    }

    /**
     * Fit the imputer to the feature column of the training data.
     *
     * @param  array $values
     * @return void
     */
    public function fit(array $values) : void
    {
        $this->mean = Average::mean($values);

        $this->stddev = sqrt(array_reduce($values, function ($carry, $value) {
            return $carry += ($value - $this->mean) ** 2;
        }, 0.0) / count($values)) + self::EPSILON;
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @return mixed
     */
    public function guess()
    {
        return $this->mean + $this->blurr
            * $this->generateGaussianValue()
            * $this->stddev;
    }

    /**
     * Generate a float value between -1 and 1.
     *
     * @return float
     */
    protected function generateGaussianValue() : float
    {
        return random_int(-1 * 1e8, 1 * 1e8) / 1e8;
    }
}
