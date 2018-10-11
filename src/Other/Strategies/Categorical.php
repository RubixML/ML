<?php

namespace Rubix\ML\Other\Strategies;

interface Categorical extends Strategy
{
    /**
     * Make a guess.
     *
     * @return string
     */
    public function guess() : string;
}
