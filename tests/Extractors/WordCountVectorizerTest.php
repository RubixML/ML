<?php

namespace Rubix\Tests\Extractors;

use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Extractors\WordCountVectorizer;
use PHPUnit\Framework\TestCase;

class WordCountVectorizerTest extends TestCase
{
    protected $extractor;

    protected $samples;

    public function setUp()
    {
        $this->samples = [
            'the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke',
            'with a dandy ubrella',
        ];

        $this->extractor = new WordCountVectorizer(50, ['quick', 'pig'], true, new Word());
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(WordCountVectorizer::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
    }

    public function test_transform_dataset()
    {
        $this->extractor->fit($this->samples);

        $samples = $this->extractor->extract($this->samples);

        $this->assertEquals([
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ], $samples);
    }
}
