<?php

namespace Rubix\ML
{
    use Rubix\ML\Exceptions\InvalidArgumentException;
    use Rubix\ML\Exceptions\RuntimeException;
    use Generator;

    use function count;
    use function is_nan;
    use function is_float;
    use function is_iterable;
    use function array_search;
    use function array_map;
    use function array_sum;
    use function exp;
    use function trigger_error;

    /**
     * Compute the argmin of the given values.
     *
     * @internal
     *
     * @template T
     * @param array<T,float|int> $values
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return T
     */
    function argmin(array $values)
    {
        $index = array_search(min($values), $values);

        if ($index === false) {
            throw new RuntimeException('Argmin is undefined for this set.');
        }

        return $index;
    }

    /**
     * Compute the argmax of the given values.
     *
     * @internal
     *
     * @template T
     * @param array<T,float|int> $values
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return T
     */
    function argmax(array $values)
    {
        $index = array_search(max($values), $values);

        if ($index === false) {
            throw new RuntimeException('Argmax is undefined for this set.');
        }

        return $index;
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
     * The logistic sigmoid function.
     *
     * @internal
     *
     * @param float $value
     * @return float
     */
    function sigmoid(float $value) : float
    {
        return 1.0 / (1.0 + exp(-$value));
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
     * Return an array of n evenly spaced numbers between minimum and maximum.
     *
     * @param float $min
     * @param float $max
     * @param int $n
     * @throws \Tensor\Exceptions\InvalidArgumentException
     * @return list<float>
     */
    function linspace(float $min, float $max, int $n) : array
    {
        if ($min > $max) {
            throw new InvalidArgumentException('Minimum must be'
                . ' less than maximum.');
        }

        if ($n < 2) {
            throw new InvalidArgumentException('Number of elements'
                . " must be greater than 1, $n given.");
        }

        $k = $n - 1;

        $interval = abs($max - $min) / $k;

        $values = [$min];

        while (count($values) < $k) {
            $values[] = end($values) + $interval;
        }

        $values[] = $max;

        return $values;
    }

    /**
     * Transpose a 2-dimensional array i.e. columns become rows and rows become columns.
     *
     * @internal
     *
     * @param array<mixed[]> $table
     * @return array<mixed[]>
     */
    function array_transpose(array $table) : array
    {
        if (count($table) < 2) {
            $columns = [];

            foreach (current($table) ?: [] as $row) {
                $columns[] = [$row];
            }

            return $columns;
        }

        return array_map(null, ...$table);
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
     * Check if an iterator contains NAN values recursively.
     *
     * @internal
     *
     * @param iterable<mixed> $values
     * @return bool
     */
    function iterator_contains_nan(iterable $values) : bool
    {
        foreach ($values as $value) {
            if (is_iterable($value)) {
                if (iterator_contains_nan($value)) {
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
