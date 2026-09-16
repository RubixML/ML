<?php

namespace Rubix\ML;

use Generator;

/**
 * Iterative
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Iterative
{
    /**
     * Return an iterable progress table from the last training session.
     *
     * @return Generator<mixed[]>
     */
    public function progress() : Generator;
}
