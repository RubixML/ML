<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class WordTest extends TestCase
{
    protected const TEXT = 'I would like to die on Mars, just not on impact. The end.';

    protected const TOKENS = [
        'I', 'would', 'like', 'to', 'die', 'on', 'Mars', 'just', 'not', 'on', 'impact',
        'The', 'end',
    ];

    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new Word();
    }

    public function test_build_tokenizer()
    {
        $this->assertInstanceOf(Word::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize()
    {
        $tokens = $this->tokenizer->tokenize(self::TEXT);

        $this->assertCount(13, $tokens);

        $this->assertEquals(self::TOKENS, $tokens);
    }
}
