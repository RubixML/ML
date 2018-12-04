<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\DeltaTfIdfTransformer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class DeltaTfIdfTransformerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Labeled([
            [1, 3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 4, 1, 0, 1],
            [0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 4, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        ], ['pos', 'neg', 'pos']);

        $this->transformer = new DeltaTfIdfTransformer();
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(DeltaTfIdfTransformer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $outcome = [
            [0.7123179275482191, 4.216395324324493, 0.0, 0.0, 0.3068528194400547, 0.0, 0.0, 0.0, 0.3068528194400547, 0.6137056388801094, 0.0, 2.8109302162163288, 0.0, 0.0, 0.0, 2.8492717101928764, 0.3068528194400547, 0.0, 0.7123179275482191],
            [0.0, 1.4054651081081644, 2.09861228866811, 0.0, 0.0, 2.8109302162163288, 2.09861228866811, 0.0, 0.0, 0.0, 0.0, 4.216395324324493, 0.0, 2.09861228866811, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7123179275482191, 0.6137056388801094, 4.216395324324493, 0.0, 0.0, 1.2274112777602189, 0.6137056388801094, 0.0, 0.0, 0.7123179275482191, 0.0, 1.4246358550964382, 0.0, 0.3068528194400547, 0.0, 0.0],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
