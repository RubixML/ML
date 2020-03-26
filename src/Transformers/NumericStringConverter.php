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
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
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

                        continue 1;
                    }
                    
                    if ($value === self::NAN_PLACEHOLDER) {
                        $value = NAN;
                    }
                }
            }
        }
    }
}
