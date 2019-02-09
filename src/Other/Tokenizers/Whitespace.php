<?php

namespace Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;

/**
 * Whitespace
 *
 * Separate each token by a user-specified delimiter such as a single
 * whitespace.
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
     * @throws \InvalidArgumentException
     */
    public function __construct(string $delimiter = ' ')
    {
        if (strlen($delimiter) < 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' at least 1 character.');
        }

        $this->delimiter = $delimiter;
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        return explode($this->delimiter, $string) ?: [];
    }
}
