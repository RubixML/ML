<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\L1Regularizer;
use PHPUnit\Framework\TestCase;

class L1RegularizerTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->transformer = new L1Regularizer();

        $this->transformer->fit(new Dataset([[1, 2, 3, 4]]));
    }

    public function test_build_l1_regularizer()
    {
        $this->assertInstanceOf(L1Regularizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            [0.0999999999, 0.1999999998, 0.29999999969999996, 0.3999999996],
            [0.4, 0.2, 0.3, 0.1],
            [0.1, 0.3, 0.2, 0.4],
        ], $data);
    }
}
