<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('RubixNumPower')]
#[CoversClass(PrincipalComponentAnalysis::class)]
class PrincipalComponentAnalysisTest extends TestCase
{
    protected Blob $generator;

    protected PrincipalComponentAnalysis $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: [0.0, 3000.0, -6.0, 25],
            stdDev: [1.0, 30.0, 0.001, 10.0]
        );

        $this->transformer = new PrincipalComponentAnalysis(2);
    }

    #[Test]
    public function fitTransform() : void
    {
        $this->assertEquals(4, $this->generator->dimensions());

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(2, $sample);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }

    #[Test]
    public function badDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new PrincipalComponentAnalysis(0);
    }

    #[Test]
    public function lossiness() : void
    {
        $this->assertNull($this->transformer->lossiness());

        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $lossiness = $this->transformer->lossiness();

        $this->assertIsFloat($lossiness);
        $this->assertGreaterThanOrEqual(0.0, $lossiness);
        $this->assertLessThanOrEqual(1.0, $lossiness);

        $full = new PrincipalComponentAnalysis(4);
        $full->fit($dataset);

        $this->assertLessThanOrEqual($lossiness, $full->lossiness());
    }

    #[Test]
    public function serializeRoundTrip() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $sample = $this->generator->generate(3)->samples();

        $expected = $sample;
        $this->transformer->transform($expected);

        $copy = unserialize(serialize($this->transformer));

        $this->assertInstanceOf(PrincipalComponentAnalysis::class, $copy);
        $this->assertTrue($copy->fitted());

        $actual = $sample;
        $copy->transform($actual);

        foreach ($actual as $i => $row) {
            $this->assertCount(count($expected[$i]), $row);

            foreach ($row as $j => $value) {
                $this->assertEqualsWithDelta($expected[$i][$j], $value, 1e-4);
            }
        }
    }
}
