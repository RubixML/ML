<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
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

    public function testDimensions() : void
    {
        $this->assertEquals(2, $this->generator->dimensions());
    }

    public function testGenerate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
        $this->assertEquals(['one', 'two'], $dataset->possibleOutcomes());
    }
}
