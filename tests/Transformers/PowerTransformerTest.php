<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Transformers\PowerTransformer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(PowerTransformer::class)]
class PowerTransformerTest extends TestCase
{
    protected PowerTransformer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new PowerTransformer();
    }

    #[Test]
    public function fitTransformReverse() : void
    {
        $dataset = Unlabeled::build(samples: [
            [1.0, 512.0, -64.0],
            [2.0, 256.0, -16.0],
            [4.0, 128.0, -4.0],
            [8.0, 64.0, -1.0],
            [16.0, 32.0, -0.25],
            [32.0, 16.0, 0.0],
            [64.0, 8.0, 0.5],
            [128.0, 4.0, 1.0],
            [256.0, 2.0, 4.0],
            [512.0, 1.0, 16.0],
        ]);

        $skews = [];

        foreach ($dataset->features() as $column) {
            $skews[] = abs(Stats::skewness($column));
        }

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $lambdas = $this->transformer->lambdas();

        $this->assertIsArray($lambdas);
        $this->assertCount(3, $lambdas);
        $this->assertContainsOnlyFloat($lambdas);

        $dataset->apply($this->transformer);

        foreach ($dataset->features() as $i => $column) {
            $this->assertLessThan($skews[$i], abs(Stats::skewness($column)));
        }

        $dataset->reverseApply($this->transformer);

        $sample = $dataset->sample(0);

        $this->assertEqualsWithDelta(1.0, $sample[0], 1e-8);
        $this->assertEqualsWithDelta(512.0, $sample[1], 1e-8);
        $this->assertEqualsWithDelta(-64.0, $sample[2], 1e-8);
    }

    #[Test]
    public function transformWithFixedLambda() : void
    {
        $transformer = new PowerTransformer(lambda: 0.0);

        $samples = Unlabeled::build(samples: [
            [0.0, -1.0],
            [1.0, 2.0],
            [2.0, 4.0],
        ]);

        $transformer->fit($samples);

        $samples->apply($transformer);

        $this->assertEqualsWithDelta(log(2.0), $samples->sample(1)[0], 1e-8);
        $this->assertEqualsWithDelta(log(3.0), $samples->sample(2)[0], 1e-8);
        $this->assertEqualsWithDelta(-1.5, $samples->sample(0)[1], 1e-8);
        $this->assertEqualsWithDelta(log(3.0), $samples->sample(1)[1], 1e-8);

        $samples->reverseApply($transformer);

        $this->assertEqualsWithDelta(1.0, $samples->sample(1)[0], 1e-8);
        $this->assertEqualsWithDelta(-1.0, $samples->sample(0)[1], 1e-8);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = Unlabeled::build(samples: [
            [1.0, 2.0],
        ])->samples();

        $this->transformer->transform($samples);
    }

    #[Test]
    public function skipsNonFinite() : void
    {
        $samples = Unlabeled::build(samples: [
            [NAN, 1.0], [NAN, 2.0],
        ]);

        $this->transformer->fit($samples);

        $samples->apply($this->transformer);

        $this->assertNan($samples[0][0]);
        $this->assertNan($samples[1][0]);

        $samples->reverseApply($this->transformer);

        $this->assertNan($samples[0][0]);
        $this->assertEqualsWithDelta(1.0, $samples[0][1], 1e-8);
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $dataset = Unlabeled::build(samples: [
            [1.0, 512.0, -64.0],
            [2.0, 256.0, -16.0],
            [4.0, 128.0, -4.0],
            [8.0, 64.0, -1.0],
            [16.0, 32.0, -0.25],
            [32.0, 16.0, 0.0],
            [64.0, 8.0, 0.5],
            [128.0, 4.0, 1.0],
            [256.0, 2.0, 4.0],
            [512.0, 1.0, 16.0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $restored = unserialize(serialize($this->transformer));

        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->lambdas(), $restored->lambdas());
    }
}
