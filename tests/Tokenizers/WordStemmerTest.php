<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Tokenizer;
use Rubix\ML\Tokenizers\WordStemmer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\WordStemmer
 */
class WordStemmerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Tokenizers\Word
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new WordStemmer('english');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(WordStemmer::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. Majority voting is likely foolish.';

        $expected = [
            'I', 'would', 'like', 'to', 'die', 'on', 'mar', 'just', 'not', 'on', 'impact',
            'major', 'vote', 'is', 'like', 'foolish',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertCount(16, $tokens);
        $this->assertEquals($expected, $tokens);
    }
}
