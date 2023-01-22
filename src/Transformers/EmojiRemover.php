<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function preg_replace;

/**
 * Emoji Remover
 *
 * This transformer removes all emojis from strings.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Stylianos Tzourelis
 */
class EmojiRemover implements Transformer
{
    /**
     * The regular expression pattern to match emojis.
     *
     * @var string
     */
    protected const PATTERN = '/[\x{1F300}-\x{1F5FF}\x{1F900}-\x{1F9FF}\x{1F600}-\x{1F64F}\x{1F680}-\x{1F6FF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}]/u';

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
        array_walk($samples, [$this, 'removeEmojis']);
    }

    /**
     * Remove emojis from a sample.
     *
     * @param list<mixed> $sample
     */
    public function removeEmojis(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_string($value)) {
                $value = preg_replace(self::PATTERN, '', $value);
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
        return 'Emoji Remover';
    }
}
