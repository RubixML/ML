<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\MultibyteTextNormalizer;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\MultibyteTextNormalizer
 */
class MultibyteTextNormalizerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\MultibyteTextNormalizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['The quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of Coke'],
            ['with a Dandy   umbrella'],
            ['Depuis qu’il avait emménagé à côté de chez elle, il y a de ça cinq ans.'],
            ['Working with emoji 🤓'],
        ]);

        $this->transformer = new MultibyteTextNormalizer();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MultibyteTextNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $outcome = [
            ['the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke'],
            ['with a dandy   umbrella'],
            ['depuis qu’il avait emménagé à côté de chez elle, il y a de ça cinq ans.'],
            ['working with emoji 🤓'],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
