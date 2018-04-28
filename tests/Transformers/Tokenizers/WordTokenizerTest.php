<?php

use Rubix\Engine\Transformers\Tokenizers\Tokenizer;
use Rubix\Engine\Transformers\Tokenizers\WordTokenizer;
use PHPUnit\Framework\TestCase;

class WordTokenizerTest extends TestCase
{
    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new WordTokenizer();
    }

    public function test_build_word_tokenizer()
    {
        $this->assertInstanceOf(WordTokenizer::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_guess_value()
    {
        $data = 'I would like to die on Mars, just not on impact.';

        $value = $this->tokenizer->tokenize($data);

        $this->assertEquals($value, ['would', 'like', 'to', 'die', 'on', 'Mars', 'just', 'not', 'on', 'impact']);
    }
}
