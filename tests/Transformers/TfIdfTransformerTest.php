<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Reversible;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\TfIdfTransformer
 */
class TfIdfTransformerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\TfIdfTransformer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled([
            [1, 3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 4, 1, 0, 1],
            [0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 4, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        ]);

        $this->transformer = new TfIdfTransformer(1.0, false);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(TfIdfTransformer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
        $this->assertInstanceOf(Reversible::class, $this->transformer);
        $this->assertInstanceOf(Persistable::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransformReverse() : void
    {
        $this->transformer->fit($this->dataset);

        $this->assertTrue($this->transformer->fitted());

        $dfs = $this->transformer->dfs();

        $this->assertIsArray($dfs);
        $this->assertCount(19, $dfs);
        $this->assertContainsOnly('int', $dfs);

        $original = clone $this->dataset;

        $this->dataset->apply($this->transformer);

        $expected = [
            [1.6931471805599454, 3.8630462173553424, 0.0, 0.0, 1.2876820724517808, 0.0, 0.0, 0.0, 1.2876820724517808, 2.5753641449035616, 0.0, 2.5753641449035616, 0.0, 0.0, 0.0, 6.772588722239782, 1.2876820724517808, 0.0, 1.6931471805599454],
            [0.0, 1.2876820724517808, 1.6931471805599454, 0.0, 0.0, 2.5753641449035616, 1.6931471805599454, 0.0, 0.0, 0.0, 0.0, 3.8630462173553424, 0.0, 1.6931471805599454, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.6931471805599454, 2.5753641449035616, 3.8630462173553424, 0.0, 0.0, 5.150728289807123, 2.5753641449035616, 0.0, 0.0, 1.6931471805599454, 0.0, 3.386294361119891, 0.0, 1.2876820724517808, 0.0, 0.0],
        ];

        $this->assertEquals($expected, $this->dataset->samples());

        $this->dataset->reverseApply($this->transformer);

        $this->assertEquals($original->samples(), $this->dataset->samples());
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
