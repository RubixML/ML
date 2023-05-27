<?php

namespace Rubix\ML\Tokenizers;

use function preg_split;

/**
 * Sentence
 *
 * This tokenizer matches sentences starting with a letter and ending with a punctuation mark.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Sentence implements Tokenizer
{
    /**
     * The regular expression to match sentences in a blob of text.
     *
     * @var string
     */
    protected const SENTENCE_REGEX = '/(?<=[\.\?\⸮\؟\!\n])(\n+|\s+|\b(?=\D))(?=[\'\"\w])/iu';

    /**
     * Tokenize a blob of text.
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array
    {
        return preg_split(self::SENTENCE_REGEX, $text, -1, PREG_SPLIT_NO_EMPTY) ?: [];
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
        return 'Sentence';
    }
}
