<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_string;
use function array_walk;
use function call_user_func;

/**
 * Text Normalizer
 *
 * This transformer converts the characters in all strings to lowercase.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TextNormalizer implements Transformer
{
    /**
     * The normalization function.
     *
     * @var callable-string
     */
    protected string $normalize;

    /**
     * @param bool $uppercase
     */
    public function __construct(bool $uppercase = false)
    {
        $this->normalize = $uppercase ? 'strtoupper' : 'strtolower';
    }

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
        array_walk($samples, [$this, 'normalize']);
    }

    /**
     * Normalize the text in a sample.
     *
     * @param list<mixed> $sample
     */
    public function normalize(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_string($value)) {
                $value = call_user_func($this->normalize, $value);
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
        return 'Text Normalizer';
    }
}
