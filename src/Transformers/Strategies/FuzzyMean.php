<?php

namespace Rubix\Engine\Transformers\Strategies;

use MathPHP\Statistics\Average;
use MathPHP\Statistics\Descriptive;
use InvalidArgumentException;

class FuzzyMean implements Continuous
{
    /**
     * The amount of gaussian noise by ratio of the variance to add to the mean.
     *
     * @var float
     */
    protected $fuzz;

    /**
     * The number of decimal places of precision to use when generating noise.
     *
     * @var int
     */
    protected $precision;

    /**
     * @param  float  $fuzz
     * @param  int  $precision
     * @return void
     */
    public function __construct($fuzz = 0.3, $precision = 3)
    {
        if ($fuzz < 0.0 || $fuzz > 1.0) {
            throw new InvalidArgumentException('The ratio of subsamples must be a float between 0 and 1.');
        }

        $this->fuzz = $fuzz;
        $this->precision = $precision;
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @param  array  $values
     * @return mixed
     */
    public function guess(array $values)
    {
        return Average::mean($values) + $this->generateGaussianValue() * $this->fuzz * Descriptive::standardDeviation($values);
    }

    /**
     * Generate a float value between -1 and 1.
     *
     * @return float
     */
    protected function generateGaussianValue() : float
    {
        $scale = pow(10, $this->precision);

        return random_int(-1 * $scale, 1 * $scale) / $scale;
    }
}
