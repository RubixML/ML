<?php

namespace Rubix\ML
{
    /**
     * Compute the argmin of the given values.
     *
     * @param array<int|float> $values
     * @return mixed
     */
    function argmin(array $values)
    {
        return array_search(min($values), $values);
    }

    /**
     * Compute the argmax of the given values.
     *
     * @param array<int|float> $values
     * @return mixed
     */
    function argmax(array $values)
    {
        return array_search(max($values), $values);
    }

    /**
     * Compute the log of the sum of exponential values.
     *
     * @param array<int|float> $values
     * @return float
     */
    function logsumexp(array $values) : float
    {
        return log(array_sum(array_map('exp', $values)));
    }

    /**
     * Transpose a 2-dimensional array i.e. rotate the data table.
     *
     * @param array[] $table
     * @return array[]
     */
    function array_transpose(array $table) : array
    {
        $m = count($table);

        switch (true) {
            case $m > 1:
                return array_map(null, ...$table);

            case $m === 1:
                $row = reset($table) ?: [];

                $n = count($row);

                $columns = [];

                for ($i = 0; $i < $n; ++$i) {
                    $columns[] = [$row[$i]];
                }

                return $columns;
            
            default:
                return $table;
        }
    }
}
