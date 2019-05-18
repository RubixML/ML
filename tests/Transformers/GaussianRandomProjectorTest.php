<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\GaussianRandomProjector;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class GaussianRandomProjectorTest extends TestCase
{
    protected $transformer;

    protected $generator;

    public function setUp()
    {
        $this->generator = new Blob(array_fill(0, 10, 0.), 3.);

        $this->transformer = new GaussianRandomProjector(5);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(GaussianRandomProjector::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_estimate_min_dimensions()
    {
        $this->assertEquals(663, GaussianRandomProjector::minDimensions(1000000, 0.5));
    }

    public function test_fit_transform()
    {
        $this->assertCount(10, $this->generator->generate(1)->row(0));

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->row(0);

        $this->assertCount(5, $sample);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
