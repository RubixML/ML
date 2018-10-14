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
        $train = $this->generator->generate(30);
        $test = $this->generator->generate(1);

        $outcomes = ['a', 'b', 'c', 'd', 'e'];

        $this->transformer->fit($train);

        $test->apply($this->transformer);

        $sample = $test->row(0);

        $this->assertContains($sample[0], $outcomes);
        $this->assertContains($sample[1], $outcomes);
        $this->assertContains($sample[2], $outcomes);
        $this->assertContains($sample[3], $outcomes);
    }

    public function test_transform_unfitted()
    {
        $dataset = $this->generator->generate(1);

        $this->expectException(RuntimeException::class);

        $dataset->apply($this->transformer);
    }
}
