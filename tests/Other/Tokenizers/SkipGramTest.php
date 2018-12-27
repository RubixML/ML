<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\SkipGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class SkipGramTest extends TestCase
{
    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new SkipGram(2, 2);
    }

    public function test_build_tokenizer()
    {
        $this->assertInstanceOf(SkipGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize()
    {
        $data = 'I would like to die on Mars, just not on impact.';

        $tokens = $this->tokenizer->tokenize($data);

        $this->assertCount(25, $tokens);

        $this->assertEquals(['I would', 'I like', 'I to', 'would like', 'would to', 'would die',
            'like to', 'like die', 'like on', 'to die', 'to on', 'to Mars', 'die on', 'die Mars',
            'die just', 'on Mars', 'on just', 'on not', 'Mars just', 'Mars not', 'Mars on',
            'just not', 'just on', 'just impact', 'on impact'], $tokens);
    }
}
