<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
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
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $dataset->apply($this->transformer);

        $this->assertEquals(2, $dataset->numColumns());
    }

    public function test_transform_unfitted()
    {
        $dataset = $this->generator->generate(1);

        $this->expectException(RuntimeException::class);

        $dataset->apply($this->transformer);
    }
}
