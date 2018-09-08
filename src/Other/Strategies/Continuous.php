<?php

namespace Rubix\ML\Other\Strategies;

interface Continuous extends Strategy
{
    /**
     * Make a continuous guess.
     *
     * @return float
     */
    public function guess() : float;
}
