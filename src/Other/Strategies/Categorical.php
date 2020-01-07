<?php

namespace Rubix\ML\Other\Strategies;

interface Categorical extends Strategy
{
    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param string[] $values
     */
    public function fit(array $values) : void;

    /**
     * Make a guess.
     *
     * @return string
     */
    public function guess() : string;
}
