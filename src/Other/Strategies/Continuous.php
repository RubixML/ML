<?php

namespace Rubix\ML\Other\Strategies;

interface Continuous extends Strategy
{
    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param (int|float)[] $values
     */
    public function fit(array $values) : void;

    /**
     * Make a guess.
     *
     * @return int|float
     */
    public function guess();
}
