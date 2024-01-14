<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function gettype;

/**
 * Stop Word Filter
 *
 * Removes user-specified words from any categorical feature columns including blobs of text.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class StopWordFilter extends RegexFilter
{
    /**
     * @param string[] $stopWords
     * @throws InvalidArgumentException
     */
    public function __construct(array $stopWords = [])
    {
        $patterns = [];

        foreach ($stopWords as &$word) {
            if (!is_string($word) or empty($word)) {
                throw new InvalidArgumentException('Stop word must be a'
                    . ' non-empty string, ' . gettype($word) . ' found.');
            }

            $word = preg_quote($word, '/');
        }

        if (!empty($stopWords)) {
            $patterns[] = sprintf('/\b(%s)\b/u', implode('|', $stopWords));
        }

        parent::__construct($patterns);
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
        return 'Stop Word Filter';
    }
}
