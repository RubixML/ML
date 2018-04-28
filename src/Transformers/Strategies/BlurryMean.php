<?php

namespace Rubix\Engine\Transformers\Strategies;

use MathPHP\Statistics\Average;
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
     * @param  float  $blurr
     * @return void
     */
    public function __construct(float $blurr = 0.2)
    {
        if ($blurr < 0.0 || $blurr > 1.0) {
            throw new InvalidArgumentException('Blurr factor must be between 0 and 1.');
        }

        $this->blurr = $blurr;
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @param  array  $values
     * @return mixed
     */
    public function guess(array $values)
    {
        $mean = Average::mean($values);

        $stddev = sqrt(array_reduce($values, function ($carry, $value) use ($mean) {
            return $carry += ($value - $mean) ** 2;
        }, 0.0) / count($values)) + self::EPSILON;

        return $mean + ($this->blurr * $this->generateGaussianValue() * $stddev);
    }

    /**
     * Generate a float value between -1 and 1.
     *
     * @return float
     */
    protected function generateGaussianValue() : float
    {
        $scale = pow(10, 8);

        return random_int(-1 * $scale, 1 * $scale) / $scale;
    }
}
