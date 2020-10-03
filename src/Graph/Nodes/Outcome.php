<?php

namespace Rubix\ML\Graph\Nodes;

use Stringable;

interface Outcome extends Decision, Stringable
{
    /**
     * Return the outcome of a decision.
     *
     * @return int|float|string
     */
    public function outcome();
}
