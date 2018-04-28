<?php

namespace Rubix\Engine\Transformers\Strategies;

interface Strategy
{
    const EPSILON = 1e-8;

    /**
     * Make a guess at a missing value.
     *
     * @param  array  $values
     * @return mixed
     */
    public function guess(array $values);
}
