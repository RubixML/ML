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
interface Outcome extends Decision, BinaryNode, Stringable
{
    /**
     * Return the outcome of the decision.
     *
     * @return string|int|float
     */
    public function outcome();
}
