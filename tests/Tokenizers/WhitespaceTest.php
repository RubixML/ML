<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Tokenizer;
use Rubix\ML\Tokenizers\Whitespace;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\Whitespace
 */
class WhitespaceTest extends TestCase
{
    /**
     * @var \Rubix\ML\Tokenizers\Whitespace
     */
    protected $tokenizer;

    protected function setUp() : void
    {
        $this->tokenizer = new Whitespace();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Whitespace::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I', 'would', 'like', 'to', 'die', 'on', 'Mars,', 'just', 'not', 'on', 'impact.',
            'The', 'end.',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(13, $tokens);
    }
}
