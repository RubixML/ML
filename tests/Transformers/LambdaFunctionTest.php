<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\LambdaFunction;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(LambdaFunction::class)]
class LambdaFunctionTest extends TestCase
{
    protected LambdaFunction $transformer;

    protected function setUp() : void
    {
        $callback = function (&$sample, $index, $context) {
            $sample = [$index, array_sum($sample), $context];
        };

        $this->transformer = new LambdaFunction(callback: $callback, context: 'context');
    }

    public function testTransform() : void
    {
        $dataset = new Unlabeled(samples: [
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            [0, 10, 'context'], [1, 100, 'context'], [2, 1000, 'context'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
