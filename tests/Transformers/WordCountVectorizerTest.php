<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\WordCountVectorizer;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\WordCountVectorizer
 */
class WordCountVectorizerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\WordCountVectorizer
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

        $this->transformer = new WordCountVectorizer(50, 1, PHP_INT_MAX, new Word());
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
        $this->transformer->fit($this->dataset);

        $this->assertTrue($this->transformer->fitted());

        $vocabulary = current($this->transformer->vocabularies());

        $this->assertIsArray($vocabulary);
        $this->assertCount(20, $vocabulary);
        $this->assertContainsOnly('string', $vocabulary);

        $this->dataset->apply($this->transformer);

        $outcome = [
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }

    /**
     * @test
     */
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->dataset->samples();

        $this->transformer->transform($samples);
    }
}
