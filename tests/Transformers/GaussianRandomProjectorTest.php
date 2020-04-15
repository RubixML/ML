<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\GaussianRandomProjector;
use PHPUnit\Framework\TestCase;
use RuntimeException;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\GaussianRandomProjector
 */
class GaussianRandomProjectorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;
    
    /**
     * @var \Rubix\ML\Transformers\GaussianRandomProjector
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob(array_fill(0, 10, 0.0), 3.0);

        $this->transformer = new GaussianRandomProjector(5);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GaussianRandomProjector::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }
    
    /**
     * @test
     */
    public function minDimensions() : void
    {
        $this->assertEquals(663, GaussianRandomProjector::minDimensions(1000000, 0.5));
    }
    
    /**
     * @test
     */
    public function fitTransform() : void
    {
        $this->assertCount(10, $this->generator->generate(1)->sample(0));

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(5, $sample);
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
