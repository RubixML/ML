<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\OneHotEncoder;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class OneHotEncoderTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ]);

        $this->transformer = new OneHotEncoder();
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(OneHotEncoder::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->transformer->fit($this->dataset);

        $this->assertTrue($this->transformer->fitted());

        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->dataset->samples();

        $this->transformer->transform($samples);
    }
}
