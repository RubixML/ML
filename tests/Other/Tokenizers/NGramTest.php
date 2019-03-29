<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\NGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class NGramTest extends TestCase
{
    protected const TEXT = 'I would like to die on Mars, just not on impact. The end.';

    protected const TOKENS = [
        'I', 'I would', 'would', 'would like', 'like', 'like to', 'to', 'to die', 'die',
        'die on', 'on', 'on Mars', 'Mars', 'Mars just', 'just', 'just not', 'not', 'not on',
        'on', 'on impact', 'impact', 'The', 'The end', 'end',
    ];

    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new NGram(1, 2);
    }

    public function test_build_tokenizer()
    {
        $this->assertInstanceOf(NGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize()
    {
        $tokens = $this->tokenizer->tokenize(self::TEXT);

        $this->assertCount(24, $tokens);

        $this->assertEquals(self::TOKENS, $tokens);
    }
}
