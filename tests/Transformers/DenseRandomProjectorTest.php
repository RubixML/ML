<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\DenseRandomProjector;
use PHPUnit\Framework\TestCase;
use RuntimeException;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\DenseRandomProjector
 */
class DenseRandomProjectorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;
    
    /**
     * @var \Rubix\ML\Transformers\DenseRandomProjector
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob(array_fill(0, 10, 0.0), 3.0);

        $this->transformer = new DenseRandomProjector(3);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(DenseRandomProjector::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
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

        $this->assertCount(3, $sample);
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
