<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\TokenHashingVectorizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\TokenHashingVectorizer
 */
class TokenHashingVectorizerTest extends TestCase
{
    /**
     * @var TokenHashingVectorizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->transformer = new TokenHashingVectorizer(20, new Word(), 'crc32');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(TokenHashingVectorizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $dataset = Unlabeled::quick([
            ['the quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of coke'],
            ['with a dandy umbrella'],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            [0, 1, 1, 0, 1, 1, 0, 4, 0, 1, 2, 1, 0, 0, 1, 1, 3, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
