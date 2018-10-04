<?php

namespace Rubix\ML\Tests\Transformers;

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
            [0.47712125471966244, 0.5282737771670437, 0.0, 0.0, 0.17609125905568124, 0.0, 0.0, 0.0, 0.17609125905568124, 0.3521825181113625, 0.0, 0.3521825181113625, 0.0, 0.0, 0.0, 1.9084850188786497, 0.17609125905568124, 0.0, 0.47712125471966244],
            [0.0, 0.17609125905568124, 0.47712125471966244, 0.0, 0.0, 0.3521825181113625, 0.47712125471966244, 0.0, 0.0, 0.0, 0.0, 0.5282737771670437, 0.0, 0.47712125471966244, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.47712125471966244, 0.3521825181113625, 0.5282737771670437, 0.0, 0.0, 0.704365036222725, 0.3521825181113625, 0.0, 0.0, 0.47712125471966244, 0.0, 0.9542425094393249, 0.0, 0.17609125905568124, 0.0, 0.0],
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
