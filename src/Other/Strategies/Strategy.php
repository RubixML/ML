<?php

namespace Rubix\ML\Other\Strategies;

interface Strategy
{
    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param (string|int|float)[] $values
     */
    public function fit(array $values) : void;

    /**
     * Make a guess.
     *
     * @return mixed
     */
    public function guess();
}
