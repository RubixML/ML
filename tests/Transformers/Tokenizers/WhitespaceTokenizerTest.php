<?php

use Rubix\Engine\Transformers\Tokenizers\Tokenizer;
use Rubix\Engine\Transformers\Tokenizers\WhitespaceTokenizer;
use PHPUnit\Framework\TestCase;

class WhitespaceTokenizerTest extends TestCase
{
    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new WhitespaceTokenizer();
    }

    public function test_build_whitespace_tokenizer()
    {
        $this->assertInstanceOf(WhitespaceTokenizer::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_guess_value()
    {
        $data = 'I would like to die on Mars, just not on impact.';

        $value = $this->tokenizer->tokenize($data);

        $this->assertEquals($value, ['I', 'would', 'like', 'to', 'die', 'on', 'Mars,', 'just', 'not', 'on', 'impact.']);
    }
}
