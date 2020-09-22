<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\MinMaxNormalizer;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\MinMaxNormalizer
 */
class MinMaxNormalizerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Transformers\MinMaxNormalizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob([0.0, 3000.0, -6.0], [1.0, 30.0, 0.001]);

        $this->transformer = new MinMaxNormalizer(0.0, 1.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MinMaxNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitUpdateTransform() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->transformer->update($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $minimums = $this->transformer->minimums();

        $this->assertIsArray($minimums);
        $this->assertCount(3, $minimums);

        $maximums = $this->transformer->maximums();

        $this->assertIsArray($maximums);
        $this->assertCount(3, $maximums);

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(3, $sample);

        $this->assertEqualsWithDelta(0.5, $sample[0], 1);
        $this->assertEqualsWithDelta(0.5, $sample[1], 1);
        $this->assertEqualsWithDelta(0.5, $sample[2], 1);
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
}
