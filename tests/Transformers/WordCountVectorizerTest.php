<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\WordCountVectorizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\WordCountVectorizer
 */
class WordCountVectorizerTest extends TestCase
{
    /**
     * @var WordCountVectorizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->transformer = new WordCountVectorizer(50, 1, 1.0, new Word());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(WordCountVectorizer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $dataset = Unlabeled::quick([
            ['the quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of coke'],
            ['with a dandy umbrella'],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $vocabulary = current($this->transformer->vocabularies() ?? []);

        $this->assertIsArray($vocabulary);
        $this->assertCount(20, $vocabulary);
        $this->assertContainsOnly('string', $vocabulary);

        $dataset->apply($this->transformer);

        $expected = [
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
