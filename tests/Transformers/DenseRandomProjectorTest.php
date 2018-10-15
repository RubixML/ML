<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\DenseRandomProjector;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class DenseRandomProjectorTest extends TestCase
{
    protected $transformer;

    protected $generator;

    public function setUp()
    {
        $this->generator = new Blob(array_fill(0, 10, 0.), 3.);

        $this->transformer = new DenseRandomProjector(3);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(DenseRandomProjector::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->assertCount(10, $this->generator->generate(1)->row(0));

        $this->transformer->fit($this->generator->generate(30));

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->row(0);

        $this->assertCount(3, $sample);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->generator->generate(1)
            ->apply($this->transformer);
    }
}
