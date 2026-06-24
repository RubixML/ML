<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Labeled;

#[Group('Generators')]
#[CoversClass(Circle::class)]
class CircleTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected const array CENTER = [5.0, 5.0];

    protected Circle $generator;

    protected function setUp() : void
    {
        $center = NumPower::array(self::CENTER)->toArray();

        $this->generator = new Circle(
            x: $center[0],
            y: $center[1],
            scale: 10.0,
            noise: 0.1
        );
    }

    #[Test]
    #[TestDox('Returns dimensions')]
    public function dimensions() : void
    {
        self::assertEquals(2, $this->generator->dimensions());
    }

    #[Test]
    #[TestDox('Generates a labeled dataset')]
    public function generate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        self::assertInstanceOf(Labeled::class, $dataset);
        self::assertInstanceOf(Dataset::class, $dataset);

        self::assertCount(self::DATASET_SIZE, $dataset);
        self::assertSame([self::DATASET_SIZE, 2], $dataset->shape());

        $samples = NumPower::array($dataset->samples());
        $labels = NumPower::array($dataset->labels());

        self::assertInstanceOf(NDArray::class, $samples);
        self::assertInstanceOf(NDArray::class, $labels);
        self::assertSame([self::DATASET_SIZE, 2], $samples->shape());
        self::assertSame([self::DATASET_SIZE], $labels->shape());
    }
}
