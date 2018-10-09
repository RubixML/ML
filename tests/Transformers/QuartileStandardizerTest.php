<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\QuartileStandardizer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class QuartileStandardizerTest extends TestCase
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

        $this->transformer = new QuartileStandardizer();
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(QuartileStandardizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_transform()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [-0.3939393938996021, -0.06040268456173145, -0.13705583755649461, -0.015151515151132538],
            [0.0, 0.0, 0.0, 0.0],
            [0.6060606059993878, 0.9395973154047115, 0.862944162392744, 0.9848484848236149],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
