<?php

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\TfIdfTransformer;
use PHPUnit\Framework\TestCase;

class TfIdfTransformerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
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

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->transform($this->transformer);

        $this->assertEquals([
            [0.4771212503767177, 0.5282737706526266, 0, 0, 0.17609125688420885, 0, 0, 0, 0.17609125688420885, 0.3521825137684177, 0, 0.3521825137684177, 0, 0, 0, 1.9084850015068708, 0.17609125688420885, 0, 0.4771212503767177],
            [0, 0.17609125688420885, 0.4771212503767177, 0, 0, 0.3521825137684177, 0.4771212503767177, 0, 0, 0, 0, 0.5282737706526266, 0, 0.4771212503767177, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.4771212503767177, 0.3521825137684177, 0.5282737706526266, 0, 0, 0.7043650275368354, 0.3521825137684177, 0, 0, 0.4771212503767177, 0, 0.9542425007534354, 0, 0.17609125688420885, 0, 0],
        ], $this->dataset->samples());
    }
}
