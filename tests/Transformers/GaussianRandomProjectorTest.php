<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\GaussianRandomProjector;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class GaussianRandomProjectorTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]);

        $this->transformer = new GaussianRandomProjector(2);
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
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals(2, $this->dataset->numColumns());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
