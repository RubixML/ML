<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\IntervalDiscretizer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(IntervalDiscretizer::class)]
class IntervalDiscretizerTest extends TestCase
{
    protected Blob $generator;

    protected IntervalDiscretizer $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: [0.0, 4.0, 0.0, -1.5],
            stdDev: [1.0, 5.0, 0.01, 10.0]
        );

        $this->transformer = new IntervalDiscretizer(bins: 5, equiWidth: false);
    }

    public function testFitTransform() : void
    {
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $intervals = $this->transformer->intervals();

        $this->assertIsArray($intervals);
        $this->assertCount(4, $intervals);
        $this->assertContainsOnlyArray($intervals);

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(4, $sample);

        $expected = ['a', 'b', 'c', 'd', 'e'];

        $this->assertContains($sample[0], $expected);
        $this->assertContains($sample[1], $expected);
        $this->assertContains($sample[2], $expected);
        $this->assertContains($sample[3], $expected);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
