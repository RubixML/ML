<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\NGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class NGramTest extends TestCase
{
    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new NGram(2);
    }

    public function test_build_tokenizer()
    {
        $this->assertInstanceOf(NGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize()
    {
        $data = 'I would like to die on Mars, just not on impact.';

        $tokens = $this->tokenizer->tokenize($data);

        $this->assertEquals(['I would', 'would like', 'like to', 'to die', 'die on', 'on Mars', 'Mars just',
            'just not', 'not on', 'on impact'], $tokens);
    }
}
