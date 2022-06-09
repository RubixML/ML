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
     * <<<<<<< HEAD
     * Return the outcome of the decision.
     *
     * @return string|int|float
     *                          =======
     *                          Return the outcome of the decision, depends on the actual class.
     *
     * @return mixed
     *               >>>>>>> 2.1
     */
    public function outcome();
}
