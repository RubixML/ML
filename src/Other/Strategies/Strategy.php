<?php

namespace Rubix\ML\Other\Strategies;

interface Strategy
{
    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param array $values
     */
    public function fit(array $values) : void;
}
