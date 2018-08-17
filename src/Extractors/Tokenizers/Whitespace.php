<?php

namespace Rubix\ML\Extractors\Tokenizers;

class Whitespace implements Tokenizer
{
    /**
     * The whitespace character that delimits each token.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * @param  string  $delimiter
     * @return void
     */
    public function __construct(string $delimiter = ' ')
    {
        $this->delimiter = $delimiter;
    }

    /**
     * Tokenize a string.
     *
     * @param  string  $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        return explode($this->delimiter, $string) ?: [];
    }
}
