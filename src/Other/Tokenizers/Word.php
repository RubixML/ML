<?php

namespace Rubix\ML\Other\Tokenizers;

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
    /**
     * Tokenize a block of text.
     *
     * @param  string  $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        $tokens = [];

        preg_match_all('/\w+/u', $string, $tokens);

        return $tokens[0];
    }
}
