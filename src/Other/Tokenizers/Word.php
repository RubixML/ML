<?php

namespace Rubix\ML\Other\Tokenizers;

use function preg_match_all;

/**
 * Word
 *
 * This tokenizer matches words with 1 or more characters.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Word implements Tokenizer
{
    const WORD_REGEX = '/\w+/u';

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        $words = [];

        preg_match_all(self::WORD_REGEX, $string, $words);

        return $words[0];
    }
}
