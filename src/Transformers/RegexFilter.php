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
 * References:
 * [1] J. Gruber. (2009). A Liberal, Accurate Regex Pattern for Matching URLs.
 * [2] J. Gruber. (2010). An Improved Liberal, Accurate Regex Pattern for Matching URLs.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RegexFilter implements Transformer
{
    /**
     * The default URL matching pattern.
     *
     * @var string
     */
    public const URL = self::GRUBER_1;

    /**
     * The original Gruber URL matching pattern.
     *
     * @var string
     */
    public const GRUBER_1 = '%\b(([\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))%s';

    /**
     * The improved Gruber URL matching pattern.
     *
     * @var string
     */
    public const GRUBER_2 = '%(?xi)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))%s';

    /**
     * A pattern to match email addresses.
     *
     * @var string
     */
    public const EMAIL = '/[a-z0-9_\-\+\.]+@[a-z0-9\-]+\.([a-z]{2,4})(?:\.[a-z]{2})?/i';

    /**
     * A pattern to match Twitter-style mentions (ex. @RubixML).
     *
     * @var string
     */
    public const MENTION = '/(@\w+)/';

    /**
     * A pattern to match Twitter-style hashtags (ex. #MachineLearning).
     *
     * @var string
     */
    public const HASHTAG = '/(#\w+)/';

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
    public function __construct(array $patterns)
    {
        if (empty($patterns)) {
            throw new InvalidArgumentException('Must specify at least'
                . ' 1 regex pattern.');
        }

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
                    $value = preg_replace($this->patterns, '', $value);
                }
            }
        }
    }
}
