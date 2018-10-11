<?php

namespace Rubix\ML\Other\Strategies;

interface Continuous extends Strategy
{
    /**
     * Make a guess.
     *
     * @return float
     */
    public function guess() : float;
}
