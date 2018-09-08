<?php

namespace Rubix\ML\Other\Strategies;

interface Categorical extends Strategy
{
    /**
     * Make a categorical guess.
     *
     * @return string
     */
    public function guess() : string;
}
