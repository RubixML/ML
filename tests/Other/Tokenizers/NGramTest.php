<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\NGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class NGramTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\NGram
     */
    protected $tokenizer;

    public function setUp() : void
    {
        $this->tokenizer = new NGram(1, 2);
    }

    public function test_build_tokenizer() : void
    {
        $this->assertInstanceOf(NGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I', 'I would', 'would', 'would like', 'like', 'like to', 'to', 'to die', 'die',
            'die on', 'on', 'on Mars', 'Mars', 'Mars just', 'just', 'just not', 'not', 'not on',
            'on', 'on impact', 'impact', 'The', 'The end', 'end',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertCount(24, $tokens);
        $this->assertEquals($expected, $tokens);
    }
}
