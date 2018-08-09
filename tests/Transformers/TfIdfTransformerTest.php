<?php

namespace Rubix\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\TfIdfTransformer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class TfIdfTransformerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    protected $outcome;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [1, 3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 4, 1, 0, 1],
            [0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 4, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        ]);

        $this->transformer = new TfIdfTransformer();

        $this->outcome = [
            [0.4771212518243659, 0.5282737749955713, 0, 0, 0.1760912583318571, 0, 0, 0, 0.1760912583318571, 0.3521825166637142, 0, 0.3521825166637142, 0, 0, 0, 1.9084850072974635, 0.1760912583318571, 0, 0.4771212518243659],
            [0, 0.1760912583318571, 0.4771212518243659, 0, 0, 0.3521825166637142, 0.4771212518243659, 0, 0, 0, 0, 0.5282737749955713, 0, 0.4771212518243659, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.4771212518243659, 0.3521825166637142, 0.5282737749955713, 0, 0, 0.7043650333274284, 0.3521825166637142, 0, 0, 0.4771212518243659, 0, 0.9542425036487318, 0, 0.1760912583318571, 0, 0],
        ];
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(TfIdfTransformer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_fitted()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals($this->outcome, $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
