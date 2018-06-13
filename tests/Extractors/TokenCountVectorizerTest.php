<?php

use Rubix\Engine\Extractors\Extractor;
use Rubix\Engine\Extractors\TokenCountVectorizer;
use PHPUnit\Framework\TestCase;

class TokenCountVectorizerTest extends TestCase
{
    protected $extractor;

    protected $samples;

    public function setUp()
    {
        $this->samples = [
            'the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke',
            'with an ubrella',
        ];

        $this->extractor = new TokenCountVectorizer();
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(TokenCountVectorizer::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
    }

    public function test_transform_dataset()
    {
        $this->extractor->fit($this->samples);

        $samples = $this->extractor->extract($this->samples);

        $this->assertEquals([
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ], $samples);
    }
}
