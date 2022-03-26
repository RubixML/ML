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
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\TokenHashingVectorizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['the quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of coke'],
            ['with a dandy umbrella'],
        ]);

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
        $this->dataset->apply($this->transformer);

        $outcome = [
            [0, 1, 1, 0, 1, 1, 0, 4, 0, 1, 2, 1, 0, 0, 1, 1, 3, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
