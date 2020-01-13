<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\NGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Other\Tokenizers\NGram
 */
class NGramTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\NGram
     */
    protected $tokenizer;
    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new NGram(1, 2);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(NGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }
    
    /**
     * @test
     */
    public function tokenize() : void
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
