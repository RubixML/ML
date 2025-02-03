<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\TfIdfTransformer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TfIdfTransformer::class)]
class TfIdfTransformerTest extends TestCase
{
    protected TfIdfTransformer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new TfIdfTransformer(smoothing: 1.0, sublinear: false);
    }

    public function testFitTransformReverse() : void
    {
        $dataset = new Unlabeled(samples: [
            [1, 3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 4, 1, 0, 1],
            [0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 4, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dfs = $this->transformer->dfs();

        $this->assertIsArray($dfs);
        $this->assertCount(19, $dfs);
        $this->assertContainsOnlyInt($dfs);

        $original = clone $dataset;

        $dataset->apply($this->transformer);

        $expected = [
            [1.6931471805599454, 3.8630462173553424, 0.0, 0.0, 1.2876820724517808, 0.0, 0.0, 0.0, 1.2876820724517808, 2.5753641449035616, 0.0, 2.5753641449035616, 0.0, 0.0, 0.0, 6.772588722239782, 1.2876820724517808, 0.0, 1.6931471805599454],
            [0.0, 1.2876820724517808, 1.6931471805599454, 0.0, 0.0, 2.5753641449035616, 1.6931471805599454, 0.0, 0.0, 0.0, 0.0, 3.8630462173553424, 0.0, 1.6931471805599454, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.6931471805599454, 2.5753641449035616, 3.8630462173553424, 0.0, 0.0, 5.150728289807123, 2.5753641449035616, 0.0, 0.0, 1.6931471805599454, 0.0, 3.386294361119891, 0.0, 1.2876820724517808, 0.0, 0.0],
        ];

        $this->assertEquals($expected, $dataset->samples());

        $dataset->reverseApply($this->transformer);

        $this->assertEquals($original->samples(), $dataset->samples());
    }
}
