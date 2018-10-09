<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\LambdaFunction;
use PHPUnit\Framework\TestCase;

class LambdaFunctionTest extends TestCase
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

        $this->transformer = new LambdaFunction(function ($samples) {
            $sigmas = [];

            foreach ($samples as $sample) {
                $sigmas[] = [array_sum($sample)];
            }

            return $sigmas;
        });
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(LambdaFunction::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform()
    {
        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [10], [100], [1000],
        ], $this->dataset->samples());
    }
}
