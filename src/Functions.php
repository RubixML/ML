<?php

namespace Rubix\ML
{
    /**
     * Compute the argmin of the given values.
     *
     * @param array $values
     * @return mixed
     */
    function argmin(array $values)
    {
        return $values ? array_search(min($values), $values) : null;
    }

    /**
     * Compute the argmax of the given values.
     *
     * @param array $values
     * @return mixed
     */
    function argmax(array $values)
    {
        return $values ? array_search(max($values), $values) : null;
    }

    /**
     * Compute the log of the sum of exponential values.
     *
     * @param array $values
     * @return float|null
     */
    function logsumexp(array $values) : ?float
    {
        return $values ? log(array_sum(array_map('exp', $values))) : null;
    }
}
