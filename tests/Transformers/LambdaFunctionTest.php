<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\LambdaFunction;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\LambdaFunction
 */
class LambdaFunctionTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\LambdaFunction
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
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
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(LambdaFunction::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }
    
    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [10], [100], [1000],
        ], $this->dataset->samples());
    }
}
