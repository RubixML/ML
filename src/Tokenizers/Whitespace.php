<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function explode;

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
     * @var non-empty-string
     */
    protected string $delimiter;

    /**
     * @param string $delimiter
     * @throws InvalidArgumentException
     */
    public function __construct(string $delimiter = ' ')
    {
        if (empty($delimiter)) {
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
        return explode($this->delimiter, $text);
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
        return "Whitespace (delimiter: {$this->delimiter})";
    }
}
