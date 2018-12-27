<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class WordTest extends TestCase
{
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
        $data = 'I would like to die on Mars, just not on impact.';

        $tokens = $this->tokenizer->tokenize($data);

        $this->assertCount(11, $tokens);

        $this->assertEquals(['I', 'would', 'like', 'to', 'die', 'on', 'Mars',
            'just', 'not', 'on', 'impact'], $tokens);
    }
}
