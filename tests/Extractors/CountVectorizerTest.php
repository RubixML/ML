<?php

namespace Rubix\Tests\Extractors;

use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\CountVectorizer;
use PHPUnit\Framework\TestCase;

class CountVectorizerTest extends TestCase
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

        $this->extractor = new CountVectorizer();
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(CountVectorizer::class, $this->extractor);
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
