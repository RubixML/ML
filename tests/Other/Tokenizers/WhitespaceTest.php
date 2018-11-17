<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Tokenizer;
use Rubix\ML\Other\Tokenizers\Whitespace;
use PHPUnit\Framework\TestCase;

class WhitespaceTest extends TestCase
{
    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new Whitespace();
    }

    public function test_build_tokenizer()
    {
        $this->assertInstanceOf(Whitespace::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize()
    {
        $data = 'I would like to die on Mars, just not on impact.';

        $tokens = $this->tokenizer->tokenize($data);

        $this->assertEquals(['I', 'would', 'like', 'to', 'die', 'on',
            'Mars,', 'just', 'not', 'on', 'impact.'], $tokens);
    }
}
