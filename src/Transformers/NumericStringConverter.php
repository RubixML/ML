<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

/**
 * Numeric String Converter
 *
 * Convert all numeric strings to their equivalent integer and floating point types.
 * Useful for when extracting from a source that only recognizes data as string
 * types such as CSV.
 *
 * **Note:** The string representation of the PHP constant NAN (not a number) is `NaN`.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NumericStringConverter implements Transformer
{
    /**
     * The numeric string representation of NaN.
     *
     * @var string
     */
    public const NAN_PLACEHOLDER = 'NaN';

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
     * @param list<list<mixed>> $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$value) {
                if (is_string($value)) {
                    if (is_numeric($value)) {
                        $value = (int) $value == $value
                            ? (int) $value
                            : (float) $value;

                        continue;
                    }

                    if ($value === self::NAN_PLACEHOLDER) {
                        $value = NAN;
                    }
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Numeric String Converter';
    }
}
