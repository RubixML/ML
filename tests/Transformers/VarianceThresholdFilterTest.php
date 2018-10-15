<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\VarianceThresholdFilter;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class VarianceThresholdFilterTest extends TestCase
{
    protected $transformer;

    protected $generator;

    public function setUp()
    {
        $this->generator = new Blob([0., 0., 0.], [1., 5., 0.001]);

        $this->transformer = new VarianceThresholdFilter(0.1);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(VarianceThresholdFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->transformer->fit($this->generator->generate(30));

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->row(0);

        $this->assertCount(2, $sample);
        
        $this->assertEquals(0., $sample[0], '', 3.);
        $this->assertEquals(0., $sample[1], '', 15.);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->generator->generate(1)
            ->apply($this->transformer);
    }
}
