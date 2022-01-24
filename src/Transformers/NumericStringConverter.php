<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_string;
use function is_numeric;

/**
 * Numeric String Converter
 *
 * Convert all numeric strings to their equivalent integer and floating point types.
 * Useful for when extracting from a source that only recognizes data as string
 * types such as CSV.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NumericStringConverter implements Transformer, Reversible
{
    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'convertToNumber']);
    }

    /**
     * Perform the reverse transformation to the samples.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function reverseTransform(array &$samples) : void
    {
        array_walk($samples, [$this, 'convertToString']);
    }

    /**
     * Convert numeric strings to integer and floating point numbers.
     *
     * @param list<mixed> $sample
     */
    protected function convertToNumber(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_string($value)) {
                if (is_numeric($value)) {
                    $value = (int) $value == $value
                        ? (int) $value
                        : (float) $value;

                    continue;
                }

                switch ($value) {
                    case 'NAN':
                        $value = NAN;

                        break;

                    case 'INF':
                        $value = INF;

                        break;

                    case '-INF':
                        $value = -INF;
                }
            }
        }
    }

    /**
     * Convert numbers to their numeric string representation.
     *
     * @param list<mixed> $sample
     */
    protected function convertToString(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_float($value) or is_int($value)) {
                $value = (string) $value;
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Numeric String Converter';
    }
}
