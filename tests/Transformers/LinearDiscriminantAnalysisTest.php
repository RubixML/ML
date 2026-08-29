<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('RubixNumPower')]
#[CoversClass(LinearDiscriminantAnalysis::class)]
class LinearDiscriminantAnalysisTest extends TestCase
{
    protected Agglomerate $generator;

    protected LinearDiscriminantAnalysis $transformer;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(center: [255, 0, 0], stdDev: 30.0),
                'green' => new Blob(center: [0, 128, 0], stdDev: 10.0),
                'blue' => new Blob(center: [0, 0, 255], stdDev: 20.0),
            ],
            weights: [3, 4, 3]
        );

        $this->transformer = new LinearDiscriminantAnalysis(1);
    }

    #[Test]
    public function fitTransform() : void
    {
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(3)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(1, $sample);
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

        new LinearDiscriminantAnalysis(0);
    }

    #[Test]
    public function requiresLabeledDataSet() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->transformer->fit(Unlabeled::quick($this->generator->generate(10)->samples()));
    }

    #[Test]
    public function requiresCategoricalLabels() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->transformer->fit(Labeled::quick(
            samples: $this->generator->generate(10)->samples(),
            labels: [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        ));
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
    }

    #[Test]
    public function serializeRoundTrip() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $sample = $this->generator->generate(3)->samples();

        $expected = $sample;
        $this->transformer->transform($expected);

        $copy = unserialize(serialize($this->transformer));

        $this->assertInstanceOf(LinearDiscriminantAnalysis::class, $copy);
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
