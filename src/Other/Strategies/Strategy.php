<?php

namespace Rubix\ML\Other\Strategies;

interface Strategy
{
    const EPSILON = 1e-8;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param  array  $values
     * @return void
     */
    public function fit(array $values) : void;

    /**
     * Make a guess.
     *
     * @return mixed
     */
    public function guess();
}
