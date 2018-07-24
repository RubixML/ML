<?php

namespace Rubix\ML\Other\Strategies;

use MathPHP\Probability\Distribution\Continuous\Uniform;
use InvalidArgumentException;
use RuntimeException;

/**
 * Wild Guess
 *
 * It's just what you think it is. Make a guess somewhere in between the minimum
 * and maximum values observed during fitting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WildGuess implements Continuous
{
    /**
     * The probability distribution to sample from.
     *
     * @var \MathPHP\Probability\Distribution\Continuous\Uniform|null
     */
    protected $distribution;

    /**
     * Copy the values.
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

        $this->distribution = new Uniform(min($values), max($values));
    }

    /**
     * Choose a random value between the minimum and the maximum of the fitted
     * data.
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
