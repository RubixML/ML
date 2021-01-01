<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\DenseRandomProjector;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\RuntimeException;

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
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->expectDeprecation();

        $transformer = new DenseRandomProjector(3);

        $this->assertInstanceOf(DenseRandomProjector::class, $transformer);
        $this->assertInstanceOf(Transformer::class, $transformer);
        $this->assertInstanceOf(Stateful::class, $transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $this->expectDeprecation();

        $transformer = new DenseRandomProjector(3);

        $this->assertCount(10, $this->generator->generate(1)->sample(0));

        $transformer->fit($this->generator->generate(30));

        $this->assertTrue($transformer->fitted());

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
        $this->expectDeprecation();

        $transformer = new DenseRandomProjector(3);

        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $transformer->transform($samples);
    }
}
