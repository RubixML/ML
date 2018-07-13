<?php

namespace Rubix\ML\Other\Strategies;

interface Continuous extends Strategy
{
    /**
     * Return the range of possible guesses for this strategy in a tuple.
     *
     * @return array
     */
    public function range() : array;
}
