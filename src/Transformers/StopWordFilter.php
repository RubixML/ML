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
        $patterns = [];

        foreach ($stopWords as $word) {
            if (!is_string($word)) {
                throw new InvalidArgumentException('Stop word must be a'
                    . ' string, ' . gettype($word) . ' found.');
            }

            $patterns[] = '/\b' . preg_quote($word, '/') . '\b/';
        }

        parent::__construct($patterns);
    }
}
