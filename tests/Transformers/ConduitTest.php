<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\Conduit;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Transformers\ZScaleStandardizer;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\Conduit
 */
class ConduitTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Transformers\Conduit
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob([0.0, 3000.0, -6.0], [1.0, 30.0, 0.001]);

        $this->transformer = new Conduit([
            new GaussianRandomProjector(2),
            new ZScaleStandardizer(),
        ]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Conduit::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitUpdateTransform() : void
    {
        $this->assertCount(3, $this->generator->generate(1)->sample(0));

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $this->transformer->update($this->generator->generate(10));

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(2, $sample);

        $this->assertEqualsWithDelta(0, $sample[0], 6);
        $this->assertEqualsWithDelta(0, $sample[1], 6);
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
