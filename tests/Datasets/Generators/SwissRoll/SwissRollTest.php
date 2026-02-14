<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Datasets\Generators\SwissRoll;

use Rubix\ML\Datasets\Generators\SwissRoll\SwissRoll;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use PHPUnit\Framework\TestCase;

#[Group('Generators')]
#[CoversClass(SwissRoll::class)]
class SwissRollTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected SwissRoll $generator;

    protected function setUp() : void
    {
        $this->generator = new SwissRoll(x: 0.0, y: 0.0, z: 0.0, scale: 1.0, depth: 12.0, noise: 0.3);
    }

    #[Test]
    #[TestDox('Dimensions returns 3')]
    public function testDimensions() : void
    {
        self::assertEquals(3, $this->generator->dimensions());
    }

    #[Test]
    #[TestDox('Generate returns a labeled dataset of the requested size')]
    public function testGenerate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        self::assertInstanceOf(Labeled::class, $dataset);
        self::assertInstanceOf(Dataset::class, $dataset);

        self::assertCount(self::DATASET_SIZE, $dataset);
    }
}
