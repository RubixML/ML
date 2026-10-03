<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\PolynomialExpander;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(Pipeline::class)]
class PipelineTest extends TestCase
{
    #[Test]
    public function params() : void
    {
        $transformer = new Pipeline([
            new MinMaxNormalizer(),
            new PolynomialExpander(2),
        ]);

        $expected = [
            'transformers' => [
                new MinMaxNormalizer(),
                new PolynomialExpander(2),
            ],
        ];

        $this->assertEquals($expected, $transformer->params());
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertEquals(DataType::all(), (new Pipeline([]))->compatibility());

        $pipeline = new Pipeline([
            new PolynomialExpander(2),
            new MinMaxNormalizer(),
        ]);

        $this->assertEquals([
            DataType::continuous(),
        ], $pipeline->compatibility());
    }

    #[Test]
    public function fittedInitialState() : void
    {
        $this->assertFalse((new Pipeline([new MinMaxNormalizer()]))->fitted());

        $this->assertTrue((new Pipeline([new PolynomialExpander(2)]))->fitted());

        $this->assertTrue((new Pipeline([]))->fitted());
    }

    #[Test]
    public function fitUpdatesElasticTransformers() : void
    {
        $transformer = new Pipeline([new MinMaxNormalizer(0.0, 1.0)]);

        $dataset = new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]);

        $transformer->fit($dataset);

        $this->assertTrue($transformer->fitted());

        $this->assertEqualsWithDelta([[0.0], [0.5], [1.0]], $dataset->samples(), 1e-8);
    }

    #[Test]
    public function fitTransformsThroughCompositePipeline() : void
    {
        $transformer = new Pipeline([
            new MinMaxNormalizer(0.0, 1.0),
            new PolynomialExpander(2),
        ]);

        $dataset = new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]);

        $transformer->fit($dataset);

        $expected = [
            [0.0, 0.0],
            [0.5, 0.25],
            [1.0, 1.0],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1e-8);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $transformer = new Pipeline([new MinMaxNormalizer()]);

        $this->expectException(RuntimeException::class);

        $samples = [[1.0]];

        $transformer->transform($samples);
    }

    #[Test]
    public function updateRefinesElasticTransformers() : void
    {
        $transformer = new Pipeline([new MinMaxNormalizer(0.0, 1.0)]);

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $transformer->update(new Unlabeled(samples: [
            [0.0],
            [4.0],
        ]));

        $dataset = new Unlabeled(samples: [
            [2.0],
        ]);

        $dataset->apply($transformer);

        $this->assertEqualsWithDelta(0.5, $dataset->samples()[0][0], 1e-8);
    }

    #[Test]
    public function updateLazilyFitsUnfittedTransformers() : void
    {
        $transformer = new Pipeline([new MinMaxNormalizer(0.0, 1.0)]);

        $this->assertFalse($transformer->fitted());

        $dataset = new Unlabeled(samples: [
            [0.0],
            [2.0],
        ]);

        $transformer->update($dataset);

        $this->assertTrue($transformer->fitted());

        $this->assertEqualsWithDelta([[0.0], [1.0]], $dataset->samples(), 1e-8);
    }

    #[Test]
    public function emptyTransformersAreANoOp() : void
    {
        $transformer = new Pipeline([]);

        $this->assertTrue($transformer->fitted());

        $dataset = new Unlabeled(samples: [
            [1.0, 2.0],
            [3.0, 4.0],
        ]);

        $transformer->fit($dataset);

        $dataset->apply($transformer);

        $this->assertSame([[1.0, 2.0], [3.0, 4.0]], $dataset->samples());
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $transformer = new Pipeline([new MinMaxNormalizer()]);

        $this->assertStringStartsWith('Pipeline', (string) $transformer);
    }
}
