<?php

namespace Rubix\ML
{
    /**
     * Compute the argmin of the given values.
     *
     * @param (int|float)[] $values
     * @return mixed
     */
    function argmin(array $values)
    {
        return array_search(min($values), $values);
    }

    /**
     * Compute the argmax of the given values.
     *
     * @param (int|float)[] $values
     * @return mixed
     */
    function argmax(array $values)
    {
        return array_search(max($values), $values);
    }

    /**
     * Compute the log of the sum of exponential values.
     *
     * @param (int|float)[] $values
     * @return float
     */
    function logsumexp(array $values) : float
    {
        return log(array_sum(array_map('exp', $values)));
    }

    /**
     * Compute n choose k.
     *
     * @param int $n
     * @param int $k
     * @return int
     */
    function comb(int $n, int $k) : int
    {
        return $k === 0 ? 1 : (int) (($n * comb($n - 1, $k - 1)) / $k);
    }

    /**
     * Transpose a 2-dimensional array i.e. columns become rows and rows become columns.
     *
     * @param array[] $table
     * @return array[]
     */
    function array_transpose(array $table) : array
    {
        switch (count($table)) {
            case 0:
                return $table;

            case 1:
                $columns = [];

                foreach (current($table) ?: [] as $row) {
                    $columns[] = [$row];
                }

                return $columns;

            default:
                return array_map(null, ...$table);
        }
    }

    /**
     * Emit a deprecation warning with a message.
     *
     * @param string $message
     */
    function warn_deprecated(string $message) : void
    {
        trigger_error($message, E_USER_DEPRECATED);
    }
}
