<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

/**
 * Numeric String Converter
 *
 * Convert all numeric strings into their integer and floating point Countertypes.
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
     * The placeholder string for NaN values.
     *
     * @var string
     */
    protected $placeholder;

    /**
     * @param string $placeholder
     */
    public function __construct(string $placeholder = 'NaN')
    {
        $this->placeholder = $placeholder;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
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
                switch (true) {
                    case is_string($value) and is_numeric($value):
                        $value = (int) $value == $value
                            ? (int) $value
                            : (float) $value;

                        break 1;

                    case $value === $this->placeholder:
                        $value = NAN;

                        break 1;
                }
            }
        }
    }
}
