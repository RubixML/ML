<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Bidirectional;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\MaxAbsoluteScaler;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\MaxAbsoluteScaler
 */
class MaxAbsoluteScalerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Transformers\MaxAbsoluteScaler
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob([0.0, 3000.0, -6.0], [1.0, 30.0, 0.001]);

        $this->transformer = new MaxAbsoluteScaler();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MaxAbsoluteScaler::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
        $this->assertInstanceOf(Bidirectional::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitUpdateTransform() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->transformer->update($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $maxabs = $this->transformer->maxabs();

        $this->assertIsArray($maxabs);
        $this->assertCount(3, $maxabs);

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(3, $sample);

        $this->assertEqualsWithDelta(0, $sample[0], 1);
        $this->assertEqualsWithDelta(0, $sample[1], 1);
        $this->assertEqualsWithDelta(0, $sample[2], 1);
    }

    /**
     * @test
     */
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }

    /**
     * @test
     */
    public function reverseTransform() : void
    {
        $this->transformer->fit($this->generator->generate(30));
        $samples = $expected = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
        $this->assertNotEquals($expected, $samples);

        $this->transformer->reverseTransform($samples);
        $this->assertEquals($expected, $samples);
    }

    /**
     * @test
     */
    public function reverseTransformWithReverseApply() : void
    {
        $this->transformer->fit($this->generator->generate(30));
        $samples = $this->generator->generate(1)->samples();
        $dataset = Unlabeled::fromIterator($samples);

        $dataset->apply($this->transformer);
        $this->assertNotEquals($dataset->samples(), $samples);

        $dataset->reverse($this->transformer);
        $this->assertEquals($dataset->samples(), $samples);
    }

    /**
     * @test
     */
    public function reverseTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->reverseTransform($samples);
    }
}
