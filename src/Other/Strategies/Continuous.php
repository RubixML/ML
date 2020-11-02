<?php

namespace Rubix\ML\Other\Strategies;

/**
 * Continuous
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Continuous extends Strategy
{
    /**
     * Make a guess.
     *
     * @internal
     *
     * @return int|float
     */
    public function guess();
}
