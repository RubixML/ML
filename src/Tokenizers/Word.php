<?php

namespace Rubix\ML\Tokenizers;

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
     * The regular expression to match words in a sentence.
     *
     * @var string
     */
    protected const WORD_REGEX = "/[\w'-]+/u";

    /**
     * Tokenize a blob of text.
     *
     * @internal
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array
    {
        $tokens = [];

        preg_match_all(self::WORD_REGEX, $text, $tokens);

        return $tokens[0];
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
        return 'Word';
    }
}
