<?php

namespace Rubix\ML\Metrics\Validation;

interface Validation
{
    const EPSILON = 1e-8;

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array;
}
