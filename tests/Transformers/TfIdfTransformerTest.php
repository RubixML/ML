<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\TfIdfTransformer;
use PHPUnit\Framework\TestCase;

class TfIdfTransformerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Dataset([
            [1, 3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 4, 1, 0, 1],
            [0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 4, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        ]);

        $this->transformer = new TfIdfTransformer();
    }

    public function test_build_tf_idf_transformer()
    {
        $this->assertInstanceOf(TfIdfTransformer::class, $this->transformer);
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

        $samples = $this->dataset->samples();

        $this->transformer->transform($samples);

        $this->assertTrue(true);
    }
}
