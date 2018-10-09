<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Online;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\MinMaxNormalizer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class MinMaxNormalizerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]);

        $this->transformer = new MinMaxNormalizer(0., 1.);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(MinMaxNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Online::class, $this->transformer);
    }

    public function test_transform_fitted()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [0.009998979695949393, 0.0066888878879329755, 0.015151124739106908, 0.010075502499744926],
            [0.40393837363534335, 0.06709157245169137, 0.15220696230255867, 0.025227017651260078],
            [1.0099989796959494, 1.006688887887933, 1.0151511247391067, 1.010075502499745],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
