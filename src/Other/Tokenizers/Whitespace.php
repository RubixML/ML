<?php

namespace Rubix\ML\Other\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Whitespace
 *
 * Separate each token by a user-specified delimiter such as a whitespace character.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Whitespace implements Tokenizer
{
    /**
     * The whitespace character that delimits each token.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * @param string $delimiter
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(string $delimiter = ' ')
    {
        if (strlen($delimiter) < 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' at least 1 character in length.');
        }

        $this->delimiter = $delimiter;
    }

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
        return explode($this->delimiter, $text) ?: [];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Whitespace (delimiter: {$this->delimiter})";
    }
}
