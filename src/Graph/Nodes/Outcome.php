<?php

namespace Rubix\ML\Graph\Nodes;

use Stringable;

/**
 * Outcome
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Outcome extends Decision, Stringable
{
    /**
     * Return the outcome of a decision.
     *
     * @return int|float|string
     */
    public function outcome();
}
