<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use PHPUnit\Framework\TestCase;

class MinMaxMinMaxNormalizerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Dataset([
            [50, 100, 1000],
            [40, 200, 3000],
            [29, 300, 2000],
        ]);

        $this->transformer = new MinMaxNormalizer();
    }

    public function test_build_z_scale_standardizer()
    {
        $this->assertInstanceOf(MinMaxNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $data = [
            [50, 100, 1000],
            [40, 200, 3000],
            [29, 247, 1999],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            [0.9999999995238095, 0.0, 0.0],
            [0.5238095235600907, 0.499999999975, 0.999999999995],
            [0, 0.735, 0.4995],
        ], $data);
    }
}
