<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\TargetEncoder;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TargetEncoder::class)]
class TargetEncoderTest extends TestCase
{
    protected TargetEncoder $transformer;

    protected function setUp() : void
    {
        $this->transformer = new TargetEncoder();
    }

    #[Test]
    public function constructorValidation() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new TargetEncoder(smoothing: 0.0);
    }

    #[Test]
    public function fitTransformContinuous() : void
    {
        $dataset = Labeled::build(
            samples: [
                ['red', 1.0],
                ['red', 2.0],
                ['blue', 3.0],
                ['blue', 4.0],
            ],
            labels: [10.0, 20.0, 30.0, 40.0]
        );

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $encodings = $this->transformer->encodings();

        $this->assertIsArray($encodings);
        $this->assertCount(1, $encodings);

        $globalPrior = $this->transformer->globalPrior();

        $this->assertIsArray($globalPrior);
        $this->assertCount(1, $globalPrior);
        $this->assertEqualsWithDelta(25.0, $globalPrior[0], 1e-9);

        $this->assertArrayHasKey(0, $encodings);
        $this->assertEqualsWithDelta(55.0 / 3.0, $encodings[0]['red'], 1e-9);
        $this->assertEqualsWithDelta(95.0 / 3.0, $encodings[0]['blue'], 1e-9);

        $samples = [['red', 100.0], ['blue', 200.0]];
        $this->transformer->transform($samples);

        $this->assertEqualsWithDelta([[55.0 / 3.0, 100.0], [95.0 / 3.0, 200.0]], $samples, 1e-9);
    }

    #[Test]
    public function fitTransformCategorical() : void
    {
        $dataset = Labeled::build(
            samples: [
                ['red', 'small'],
                ['red', 'small'],
                ['blue', 'large'],
                ['blue', 'large'],
            ],
            labels: ['a', 'a', 'b', 'b']
        );

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $encodings = $this->transformer->encodings();

        $this->assertIsArray($encodings);
        $this->assertCount(2, $encodings);

        $globalPrior = $this->transformer->globalPrior();

        $this->assertIsArray($globalPrior);
        $this->assertCount(2, $globalPrior);

        $this->assertArrayHasKey(0, $encodings);
        $this->assertArrayHasKey(1, $encodings);

        $this->assertEqualsWithDelta([5.0 / 6.0, 1.0 / 6.0], $encodings[0]['red'], 1e-9);
        $this->assertEqualsWithDelta([1.0 / 6.0, 5.0 / 6.0], $encodings[0]['blue'], 1e-9);

        $this->assertEqualsWithDelta([5.0 / 6.0, 1.0 / 6.0], $encodings[1]['small'], 1e-9);
        $this->assertEqualsWithDelta([1.0 / 6.0, 5.0 / 6.0], $encodings[1]['large'], 1e-9);

        $samples = [['red', 'small'], ['blue', 'large']];
        $this->transformer->transform($samples);

        $this->assertIsArray($samples[0]);
        $this->assertCount(4, $samples[0]);

        $this->assertEqualsWithDelta([5.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0], $samples[0], 1e-9);
        $this->assertEqualsWithDelta([1.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0], $samples[1], 1e-9);
    }

    #[Test]
    public function fitUnknownCategory() : void
    {
        $dataset = Labeled::build(
            samples: [
                ['red', 1.0],
                ['red', 2.0],
                ['blue', 3.0],
            ],
            labels: [10.0, 30.0, 50.0]
        );

        $this->transformer->fit($dataset);

        $samples = [['green', 5.0]];
        $this->transformer->transform($samples);

        $globalMean = (10.0 + 30.0 + 50.0) / 3.0;

        $this->assertEqualsWithDelta([[$globalMean, 5.0]], $samples, 1e-9);
    }

    #[Test]
    public function fitUnknownCategoryCategorical() : void
    {
        $dataset = Labeled::build(
            samples: [
                ['red'],
                ['red'],
                ['blue'],
            ],
            labels: ['a', 'b', 'a']
        );

        $this->transformer->fit($dataset);

        $samples = [['green']];
        $this->transformer->transform($samples);

        $globalPrior = $this->transformer->globalPrior();

        $this->assertEqualsWithDelta($globalPrior, $samples[0], 1e-9);
        $this->assertCount(2, $samples[0]);
    }

    #[Test]
    public function fitUnlabeled() : void
    {
        $dataset = new Unlabeled(samples: [['red', 'blue']]);

        $this->expectException(InvalidArgumentException::class);
        $this->transformer->fit($dataset);
    }

    #[Test]
    public function fitNoCategoricalColumns() : void
    {
        $dataset = Labeled::build(
            samples: [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            labels: [0.5, 1.5]
        );

        $this->transformer->fit($dataset);

        $encodings = $this->transformer->encodings();

        $this->assertIsArray($encodings);
        $this->assertCount(0, $encodings);

        $samples = [[1.0, 2.0]];
        $this->transformer->transform($samples);

        $this->assertEquals([[1.0, 2.0]], $samples);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);
        $samples = [['red']];
        $this->transformer->transform($samples);
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $dataset = Labeled::build(
            samples: [['red', 'blue'], ['red', 'green']],
            labels: [1.0, 2.0]
        );

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $restored = unserialize(serialize($this->transformer));

        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->encodings(), $restored->encodings());
        $this->assertEquals($this->transformer->globalPrior(), $restored->globalPrior());
        $this->assertEquals((string) $this->transformer, (string) $restored);
    }
}
