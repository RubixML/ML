<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\IntervalDiscretizer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class IntervalDiscretizerTest extends TestCase
{
    protected $transformer;

    protected $generator;

    public function setUp()
    {
        $this->generator = new Blob([0., 4., 0., -1.5], [1., 5., 0.01, 10.]);

        $this->transformer = new IntervalDiscretizer(5);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(IntervalDiscretizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $outcomes = ['a', 'b', 'c', 'd', 'e'];

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->row(0);

        $this->assertCount(4, $sample);

        $this->assertContains($sample[0], $outcomes);
        $this->assertContains($sample[1], $outcomes);
        $this->assertContains($sample[2], $outcomes);
        $this->assertContains($sample[3], $outcomes);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->generator->generate(1)
            ->apply($this->transformer);
    }
}
