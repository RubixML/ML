<?php

namespace Rubix\Tests\Extractors\Tokenizers;

use Rubix\ML\Extractors\Tokenizers\Word;
use Rubix\ML\Extractors\Tokenizers\Tokenizer;

use PHPUnit\Framework\TestCase;

class WordTest extends TestCase
{
    protected $tokenizer;

    public function setUp()
    {
        $this->tokenizer = new Word();
    }

    public function test_build_word_tokenizer()
    {
        $this->assertInstanceOf(Word::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_guess_value()
    {
        $data = 'I would like to die on Mars, just not on impact.';

        $value = $this->tokenizer->tokenize($data);

        $this->assertEquals($value, ['would', 'like', 'to', 'die', 'on', 'Mars',
            'just', 'not', 'on', 'impact']);
    }
}
