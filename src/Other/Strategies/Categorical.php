<?php

namespace Rubix\ML\Other\Strategies;

interface Categorical extends Strategy
{
    /**
     * Return the set of all possible guesses for this strategy in an array.
     *
     * @return array
     */
    public function set() : array;
}
