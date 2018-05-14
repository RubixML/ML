<?php

namespace Rubix\Engine\Estimators\Strategies;

use RuntimeException;

class Lottery implements Categorical
{
    /**
     * The unique class outcomes each having equal chance of winning lottery.
     *
     * @var array
     */
    protected $classes = [
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
            throw new RuntimeException('This strategy requires at least 1 data point.');
        }

        $this->classes = array_values(array_unique($values));
    }

    /**
     * Impute a missing value by holding a lottery. Each class has an equal chance
     * of being picked.
     *
     * @return mixed
     */
    public function guess()
    {
        return $this->classes[array_rand($this->classes)];
    }
}
