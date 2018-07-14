<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use InvalidArgumentException;
use RuntimeException;

class BlurryMean implements Continuous
{
    /**
     * The amount of gaussian noise by ratio of the standard deviation to add
     * to the guess.
     *
     * @var float
     */
    protected $blur;

    /**
     * The precomputed mean of the fitted data.
     *
     * @var float|int|null
     */
    protected $mean;

    /**
     * The precomputed standard deviation of the fitted data.
     *
     * @var float|null
     */
    protected $stddev;

    /**
     * @param  float  $blur
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $blur = 0.05)
    {
        if ($blur < 0.0 or $blur > 1.0) {
            throw new InvalidArgumentException('Blurr factor must be between 0'
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
        $r = $this->blur * $this->stddev;

        return [$this->mean - $r, $this->mean + $r];
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

        $this->mean = Average::mean($values);

        $ssd = 0.0;

        foreach ($values as $value) {
            $ssd += ($value - $this->mean) ** 2;
        }

        $this->stddev = sqrt($ssd / count($values));
    }

    /**
     * Guess a value based on the mean plus a fuzz factor of Gaussian noise.
     *
     * @throws \RuntimeException
     * @return mixed
     */
    public function guess()
    {
        if (is_null($this->mean) or is_null($this->stddev)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->mean + $this->blur
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
        return rand((int) (-1 * 1e8), (int) (1 * 1e8)) / 1e8;
    }
}
