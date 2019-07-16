<?php

namespace Rubix\ML\Graph\Nodes;

interface Outcome extends Decision
{
    /**
     * Return the outcome of a decision.
     *
     * @return int|float|string
     */
    public function outcome();
}
