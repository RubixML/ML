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
     * @var int
     */
    protected $n;

    /**
     * The popularity scores for each potential class label in the fitted data.
     *
     * @var array
     */
    protected $popularity = [
        //
    ];

    /**
     * Calculate the popularity of each unique class label in the dataset.
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

        $this->n = count($values);
        $this->popularity = array_count_values($values);
    }

    /**
     * Hold a popularity contest where the probability of winning is based on a
     * category's prior probability.
     *
     * @throws \RuntimeException
     * @return mixed
     */
    public function guess()
    {
        if (empty($this->popularity)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $random = rand(0, $this->n);

        foreach ($this->popularity as $class => $count) {
            $random -= $count;

            $temp = $class;

            if ($random < 0) {
                break 1;
            }
        }

        return $temp ?? null;
    }
}
