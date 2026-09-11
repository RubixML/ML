<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_int;
use function is_string;
use function is_numeric;
use function strtolower;

/**
 * Float Type Converter
 *
 * Convert all integer and numeric string values to their equivalent
 * floating point type. Useful for when continuous features are
 * inadvertently stored as integers by either the PHP interpreter
 * or JSON serialization, or as strings by the extraction from a
 * source that only recognizes data as string types such as CSV.
 * Both of these cases would otherwise cause the features to be
 * inferred as categorical data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FloatTypeConverter implements Transformer
{
    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
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
        array_walk($samples, [$this, 'convert']);
    }

    /**
     * Convert integers and numeric strings to their floating point equivalent.
     *
     * @param list<mixed> $sample
     */
    protected function convert(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_int($value)) {
                $value = (float) $value;

                continue;
            }

            if (is_string($value)) {
                if (is_numeric($value)) {
                    $value = (float) $value;

                    continue;
                }

                switch (strtolower($value)) {
                    case 'nan':
                        $value = NAN;

                        break;

                    case 'inf':
                        $value = INF;

                        break;

                    case '-inf':
                        $value = -INF;

                        break;
                }
            }
        }

        unset($value);
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
        return 'Float Type Converter';
    }
}
