<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Outcome
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Outcome extends Decision
{
    /**
     * Return the outcome of a decision.
     *
     * @return int|float|string
     */
    public function outcome();

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
