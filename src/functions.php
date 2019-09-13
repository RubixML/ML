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
        return array_search(min($values), $values);
    }

    /**
     * Compute the argmax of the given values.
     *
     * @param array $values
     * @return mixed
     */
    function argmax(array $values)
    {
        return array_search(max($values), $values);
    }

    /**
     * Compute the log of the sum of exponential values.
     *
     * @param array $values
     * @return float
     */
    function logsumexp(array $values) : float
    {
        return log(array_sum(array_map('exp', $values)));
    }

    /**
     * Transpose a 2-dimensional array i.e. rotate the data table.
     *
     * @param array $table
     * @return array
     */
    function transpose(array $table) : array
    {
        if (count($table) > 1) {
            return array_map(null, ...$table);
        }

        $n = count(reset($table));

        $columns = [];

        for ($i = 0; $i < $n; $i++) {
            $columns[] = array_column($table, $i);
        }

        return $columns;
    }
}
