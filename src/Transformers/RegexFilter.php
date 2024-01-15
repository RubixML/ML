<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function gettype;
use function is_string;
use function preg_replace;
use function array_walk;
use function array_values;

/**
 * Regex Filter
 *
 * Filters the text features of a dataset by matching and removing patterns from a list of regular expressions.
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
     * A pattern to match email addresses.
     *
     * @var literal-string
     */
    public const EMAIL = '/[a-z0-9_\-\+\.]+@[a-z0-9\-]+\.([a-z]{2,4})(?:\.[a-z]{2})?/i';

    /**
     * The default URL matching pattern.
     *
     * @var literal-string
     */
    public const URL = self::GRUBER_1;

    /**
     * The original Gruber URL matching pattern.
     *
     * @var literal-string
     */
    public const GRUBER_1 = '%\b(([\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))%s';

    /**
     * The improved Gruber URL matching pattern.
     *
     * @var literal-string
     */
    public const GRUBER_2 = '%(?xi)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))%s';

    /**
     * Matches consecutively repeated non word or number characters such as punctuation and special characters.
     *
     * @var literal-string
     */
    public const EXTRA_CHARACTERS = '/([^\w\s])(?=[^\w\s]*\1)/u';

    /**
     * Matches consecutively repeated words.
     *
     * @var literal-string
     */
    public const EXTRA_WORDS = '/\b(\w+)(?=\s+\1+\b)/ui';

    /**
     * Matches consecutively repeated whitespace characters.
     *
     * @var literal-string
     */
    public const EXTRA_WHITESPACE = '/\s(?=\s+)/u';

    /**
     * A pattern to match unicode emojis.
     *
     * @var literal-string
     */
    public const EMOJIS = '/[\x{1F300}-\x{1F5FF}\x{1F900}-\x{1F9FF}\x{1F600}-\x{1F64F}\x{1F680}-\x{1F6FF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}]/u';

    /**
     * A pattern to match Twitter-style mentions (ex. @RubixML).
     *
     * @var literal-string
     */
    public const MENTION = '/(@\w+)/u';

    /**
     * A pattern to match Twitter-style hashtags (ex. #MachineLearning).
     *
     * @var literal-string
     */
    public const HASHTAG = '/(#\w+)/u';

    /**
     * A list of regular expression patterns used to filter the text columns of the dataset.
     *
     * @var list<string>
     */
    protected array $patterns;

    /**
     * @param string[] $patterns
     * @throws InvalidArgumentException
     */
    public function __construct(array $patterns)
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
        if (empty($this->patterns)) {
            return;
        }

        array_walk($samples, [$this, 'filter']);
    }

    /**
     * Filter the regex patterns from the dataset.
     *
     * @param list<mixed> $sample
     */
    protected function filter(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_string($value)) {
                $value = preg_replace($this->patterns, '', $value);
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
        return 'Regex Filter';
    }
}
