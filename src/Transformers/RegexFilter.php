<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use InvalidArgumentException;

use function gettype;
use function is_string;

/**
 * Regex Filter
 *
 * Filters the text columns of a dataset by matching a list of regular expressions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RegexFilter implements Transformer
{
    public const PATTERNS = [
        'url' => '%\b(([\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))%s',
        'email' => '/[a-z0-9_\-\+\.]+@[a-z0-9\-]+\.([a-z]{2,4})(?:\.[a-z]{2})?/i',
        'mention' => '/(@\w+)/',
        'hashtag' => '/(#\w+)/',
    ];

    /**
     * A list of regular expression patterns used to filter the text columns of
     * the dataset.
     *
     * @var string[]
     */
    protected $patterns;

    /**
     * @param string[] $patterns
     * @throws \InvalidArgumentException
     */
    public function __construct(array $patterns = [])
    {
        foreach ($patterns as $pattern) {
            if (!is_string($pattern)) {
                throw new InvalidArgumentException('Pattern must be a'
                    . ' string, ' . gettype($pattern) . ' found.');
            }
        }

        $this->patterns = array_values($patterns);
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
                if (is_string($value)) {
                    $value = preg_replace($this->patterns, '', $value);
                }
            }
        }
    }
}
