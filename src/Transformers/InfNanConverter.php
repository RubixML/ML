<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_nan;
use function is_infinite;

/**
 * INF/NAN Constants Converter
 *
 * Converts INF/NAN constant values to their strings and back.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */
class InfNanConverter implements Transformer, Reversible
{
    private const NAN_STRING_VALUE = '~~NAN~~';

    private const INF_STRING_VALUE = '~~INF~~';

    private const NEGATIVE_INF_STRING_VALUE = '~~-INF~~';

    /**
     * {@inheritDoc}
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * {@inheritDoc}
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'convert']);
    }

    /**
     * {@inheritDoc}
     */
    public function reverseTransform(array &$samples) : void
    {
        array_walk($samples, [$this, 'reverseConvert']);
    }

    /**
     * Convert INF/NAN constants to their string equivalents.
     *
     * @param list<mixed> $sample
     */
    private function convert(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (!is_float($value)) {
                continue;
            }

            if (is_nan($value)) {
                $value = self::NAN_STRING_VALUE;
            } elseif (is_infinite($value) && $value > 0) {
                $value = self::INF_STRING_VALUE;
            } elseif (is_infinite($value) && $value < 0) {
                $value = self::NEGATIVE_INF_STRING_VALUE;
            }
        }
    }

    /**
     * Convert INF/NAN string values to their constant equivalents.
     *
     * @param list<mixed> $sample
     */
    private function reverseConvert(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if ($value === self::NAN_STRING_VALUE) {
                $value = NAN;
            } elseif ($value === self::INF_STRING_VALUE) {
                $value = INF;
            } elseif ($value === self::NEGATIVE_INF_STRING_VALUE) {
                $value = -INF;
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    public function __toString() : string
    {
        return 'INF/NAN constants converter';
    }
}
