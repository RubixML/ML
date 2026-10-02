<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Generators')]
#[CoversClass(Agglomerate::class)]
class AgglomerateTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected Agglomerate $generator;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'one' => new Blob(
                    center: [-5.0, 3.0],
                    stdDev: 0.2
                ),
                'two' => new Blob(
                    center: [5.0, -3.0],
                    stdDev: 0.2
                ),
            ],
            weights: [1, 0.5]
        );
    }

    #[Test]
    public function dimensions() : void
    {
        $this->assertEquals(2, $this->generator->dimensions());
    }

    #[Test]
    public function generate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
        $this->assertEquals(['one', 'two'], $dataset->possibleOutcomes());
    }

    #[Test]
    public function generateExactCountForOddSizes() : void
    {
        foreach ([3, 7, 10, 11, 13, 17, 19, 20, 50] as $n) {
            $dataset = $this->generator->generate($n);

            $this->assertSame($n, $dataset->numSamples(), "n = $n");
            $this->assertEquals(['one', 'two'], $dataset->possibleOutcomes());
        }
    }

    #[Test]
    public function generateExactCountWithThreeGenerators() : void
    {
        $agglomerate = new Agglomerate(
            generators: [
                'one' => new Blob(center: [-5.0, 3.0], stdDev: 0.2),
                'two' => new Blob(center: [5.0, -3.0], stdDev: 0.2),
                'three' => new Blob(center: [0.0, 0.0], stdDev: 0.2),
            ]
        );

        $validLabels = ['one', 'two', 'three'];

        foreach ([1, 2, 4, 6, 7, 10, 11, 17] as $n) {
            $dataset = $agglomerate->generate($n);

            $this->assertSame($n, $dataset->numSamples(), "n = $n");

            foreach ($dataset->possibleOutcomes() as $label) {
                $this->assertContains($label, $validLabels, "n = $n");
            }
        }
    }
}
