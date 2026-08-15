<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\RobustStandardizer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(RobustStandardizer::class)]
class RobustStandardizerTest extends TestCase
{
    protected Blob $generator;

    protected RobustStandardizer $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: [0.0, 3000.0, -6.0],
            stdDev: [1.0, 30.0, 0.001]
        );

        $this->transformer = new RobustStandardizer(true);
    }

    public function testFitUpdateTransformReverse() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $medians = $this->transformer->medians();

        $this->assertIsArray($medians);
        $this->assertCount(3, $medians);
        $this->assertContainsOnlyFloat($medians);

        $mads = $this->transformer->mads();

        $this->assertIsArray($mads);
        $this->assertCount(3, $mads);
        $this->assertContainsOnlyFloat($mads);

        $dataset = $this->generator->generate(1);

        $original = $dataset->sample(0);

        $dataset->apply($this->transformer);

        $sample = $dataset->sample(0);

        $this->assertCount(3, $sample);

        $this->assertEqualsWithDelta(0, $sample[0], 6);
        $this->assertEqualsWithDelta(0, $sample[1], 6);
        $this->assertEqualsWithDelta(0, $sample[2], 6);

        $dataset->reverseApply($this->transformer);

        $this->assertEqualsWithDelta($original, $dataset->sample(0), 1e-8);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
