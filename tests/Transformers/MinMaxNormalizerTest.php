<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(MinMaxNormalizer::class)]
class MinMaxNormalizerTest extends TestCase
{
    protected Blob $generator;

    protected MinMaxNormalizer $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: [0.0, 3000.0, -6.0, 1.0],
            stdDev: [1.0, 30.0, 0.001, 0.0]
        );

        $this->transformer = new MinMaxNormalizer(min: 0.0, max: 1.0);
    }

    public function testFitUpdateTransformReverse() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->transformer->update($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $minimums = $this->transformer->minimums();

        $this->assertIsArray($minimums);
        $this->assertCount(4, $minimums);

        $maximums = $this->transformer->maximums();

        $this->assertIsArray($maximums);
        $this->assertCount(4, $maximums);

        $dataset = $this->generator->generate(1);

        $original = $dataset->sample(0);

        $dataset->apply($this->transformer);

        $sample = $dataset->sample(0);

        $this->assertCount(4, $sample);

        $this->assertEqualsWithDelta(0.5, $sample[0], 1);
        $this->assertEqualsWithDelta(0.5, $sample[1], 1);
        $this->assertEqualsWithDelta(0.5, $sample[2], 1);

        $dataset->reverseApply($this->transformer);

        $this->assertEqualsWithDelta($original, $dataset->sample(0), 1e-8);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
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
