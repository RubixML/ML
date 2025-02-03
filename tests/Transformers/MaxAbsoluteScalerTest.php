<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\MaxAbsoluteScaler;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(MaxAbsoluteScaler::class)]
class MaxAbsoluteScalerTest extends TestCase
{
    protected Blob $generator;

    protected MaxAbsoluteScaler $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: [0.0, 3000.0, -6.0],
            stdDev: [1.0, 30.0, 0.001]
        );

        $this->transformer = new MaxAbsoluteScaler();
    }

    public function testFitUpdateTransformReverse() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->transformer->update($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $maxabs = $this->transformer->maxabs();

        $this->assertIsArray($maxabs);
        $this->assertCount(3, $maxabs);

        $dataset = $this->generator->generate(1);

        $original = $dataset->sample(0);

        $dataset->apply($this->transformer);

        $sample = $dataset->sample(0);

        $this->assertCount(3, $sample);

        $this->assertEqualsWithDelta(0, $sample[0], 1 + 1e-8);
        $this->assertEqualsWithDelta(0, $sample[1], 1 + 1e-8);
        $this->assertEqualsWithDelta(0, $sample[2], 1 + 1e-8);

        $dataset->reverseApply($this->transformer);

        $this->assertEqualsWithDelta($original, $dataset->sample(0), 1e-8);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }

    public function testReverseTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->reverseTransform($samples);
    }

    public function testSkipsNonFinite() : void
    {
        $samples = Unlabeled::build(samples: [
            [0.0, 3000.0, NAN, -6.0], [1.0, 30.0, NAN, 0.001]
        ]);
        $this->transformer->fit($samples);
        $this->assertNan($samples[0][2]);
        $this->assertNan($samples[1][2]);
    }
}
