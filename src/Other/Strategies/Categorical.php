<?php

namespace Rubix\ML\Other\Strategies;

/**
 * Categorical
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Categorical extends Strategy
{
    /**
     * Make a guess.
     *
     * @internal
     *
     * @return string
     */
    public function guess() : string;
}
