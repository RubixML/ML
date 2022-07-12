<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_string;
use function array_walk;

/**
 * Word Order Randomizer
 *
 * This transformer shuffles the words in a given string.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Stylianos Tzourelis
 */
class WordOrderRandomizer implements Transformer
{
    /**
     * The separator to split the string with.
     *
     * @var string
     */
    protected string $separator;

    /**
     * @param string $separator
     */
    public function __construct(string $separator = ' ')
    {
        $this->separator = $separator;
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
        array_walk($samples, [$this, 'randomize']);
    }

    /**
     * Randomize the text in a sample.
     *
     * @param list<mixed> $sample
     */
    private function randomize(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_string($value) && !empty($this->separator)) {
                $value = explode($this->separator, $value);
                shuffle($value);
                $value = implode($this->separator, $value);
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
        return 'Word Order Randomizer';
    }
}
