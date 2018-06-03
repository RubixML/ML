<?php

namespace Rubix\Engine\Transformers\Strategies;

use RuntimeException;

class PopularityContest implements Categorical
{
    /**
     * The size of the population. i.e. the sample size.
     *
     * @var int
     */
    protected $n;

    /**
     * The popularity scores for each potential class label in the fitted dataset.
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
     * @throws \RuntimeException
     * @return mixed
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new RuntimeException('This strategy requires at least 1 data'
                . ' point.');
        }

        $this->n = count($values);
        $this->popularity = array_count_values($values);
    }

    /**
     * Impute a missing value by holding a popularity contest where probability
     * of winning is based on a class's popularity among the whole.
     *
     * @return mixed
     */
    public function guess()
    {
        $random = random_int(0, $this->n);

        foreach ($this->popularity as $class => $count) {
            $random -= $count;

            $temp = $class;

            if ($random < 0) {
                break 1;
            }
        }

        return $temp;
    }
}
