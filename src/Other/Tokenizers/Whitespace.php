<?php

namespace Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;
use Stringable;

/**
 * Whitespace
 *
 * Separate each token by a user-specified delimiter such as a whitespace character.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Whitespace implements Tokenizer, Stringable
{
    /**
     * The whitespace character that delimits each token.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * @param string $delimiter
     * @throws \InvalidArgumentException
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
     * @param string $string
     * @return list<string>
     */
    public function tokenize(string $string) : array
    {
        return explode($this->delimiter, $string) ?: [];
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
