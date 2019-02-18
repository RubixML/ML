<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

/**
 * Popularity Contest
 *
 * Hold a popularity contest where the probability of winning (being guessed) is
 * based on the category's prior probability.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PopularityContest implements Categorical
{
    /**
     * The size of the population. i.e. the sample size.
     *
     * @var int|null
     */
    protected $n;

    /**
     * The popularity scores for each potential category.
     *
     * @var array|null
     */
    protected $popularity;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param array $values
     * @throws \InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fit with'
                . ' at least 1 value.');
        }

        $this->n = count($values);
        $this->popularity = array_count_values($values);
    }

    /**
     * Make a categorical guess.
     *
     * @throws \RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if ($this->n === null or !$this->popularity) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $class = current($this->popularity);

        $random = rand(0, $this->n);

        foreach ($this->popularity as $class => $count) {
            $random -= $count;

            if ($random <= 0) {
                break 1;
            }
        }

        return (string) $class;
    }
}
