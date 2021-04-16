<?php

namespace Rubix\ML
{
    use Generator;

    /**
     * Compute the argmin of the given values.
     *
     * @internal
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
     * @internal
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
     * @internal
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
     * @internal
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
     * @internal
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
     * Unset the given indices from an array.
     *
     * @internal
     *
     * @param mixed[] $values
     * @param (string|int)[] $indices
     */
    function array_unset(array &$values, array $indices) : void
    {
        foreach ($indices as $index) {
            unset($values[$index]);
        }
    }

    /**
     * Check if a multidimensional array contains NAN values recursively.
     *
     * @internal
     *
     * @param mixed[] $values
     * @return bool
     */
    function array_contains_nan(array $values) : bool
    {
        foreach ($values as $value) {
            if (is_array($value)) {
                if (array_contains_nan($value)) {
                    return true;
                }
            }

            if (is_float($value)) {
                if (is_nan($value)) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Return the first element of an iterator.
     *
     * @internal
     *
     * @param iterable<mixed> $iterator
     * @return mixed
     */
    function iterator_first(iterable $iterator)
    {
        foreach ($iterator as $element) {
            return $element;
        }
    }

    /**
     * Map a callback function over the elements of an iterator.
     *
     * @internal
     *
     * @param iterable<mixed> $iterator
     * @param callable $callback
     * @return Generator<mixed>
     */
    function iterator_map(iterable $iterator, callable $callback) : Generator
    {
        foreach ($iterator as $value) {
            yield $callback($value);
        }
    }

    /**
     * Filter the elements of an iterator using a callback.
     *
     * @internal
     *
     * @param iterable<mixed> $iterator
     * @param callable $callback
     * @return Generator<mixed>
     */
    function iterator_filter(iterable $iterator, callable $callback) : Generator
    {
        foreach ($iterator as $value) {
            if ($callback($value)) {
                yield $value;
            }
        }
    }

    /**
     * Emit a deprecation warning with a message.
     *
     * @internal
     *
     * @param string $message
     */
    function warn_deprecated(string $message) : void
    {
        trigger_error($message, E_USER_DEPRECATED);
    }
}
