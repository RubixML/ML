<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Stringable;

interface Metric extends Stringable
{
    /**
     * Return a tuple of the min and max score for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple;
}
