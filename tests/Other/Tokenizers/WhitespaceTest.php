<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Tokenizer;
use Rubix\ML\Other\Tokenizers\Whitespace;
use PHPUnit\Framework\TestCase;

class WhitespaceTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\Whitespace
     */
    protected $tokenizer;

    public function setUp() : void
    {
        $this->tokenizer = new Whitespace();
    }

    public function test_build_tokenizer() : void
    {
        $this->assertInstanceOf(Whitespace::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I', 'would', 'like', 'to', 'die', 'on', 'Mars,', 'just', 'not', 'on', 'impact.',
            'The', 'end.',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertCount(13, $tokens);
        $this->assertEquals($expected, $tokens);
    }
}
