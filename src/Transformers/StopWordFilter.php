<?php

namespace Rubix\ML\Transformers;

use InvalidArgumentException;

use function gettype;

/**
 * Stop Word Filter
 *
 * Removes user-specified words from any categorical feature columns including blobs
 * of text.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class StopWordFilter extends RegexFilter
{
    /**
     * @param string[] $stopWords
     * @throws \InvalidArgumentException
     */
    public function __construct(array $stopWords = [])
    {
        foreach ($stopWords as &$word) {
            if (!is_string($word) or empty($word)) {
                throw new InvalidArgumentException('Stop word must be a'
                    . 'non-empty string, ' . gettype($word) . ' found.');
            }

            $word = preg_quote($word, '/');
        }

        $pattern = sprintf('/\b(%s)\b/u', implode('|', $stopWords));

        parent::__construct([$pattern]);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Stop Word Filter';
    }
}
