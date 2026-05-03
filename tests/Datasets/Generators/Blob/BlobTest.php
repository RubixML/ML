<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators\Blob;

use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob\Blob;
use Rubix\ML\Datasets\Generators\Generator;

#[Group('Generators')]
#[CoversClass(Blob::class)]
class BlobTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected const array CENTER = [0.0, 0.0, 0.0];

    protected Blob $generator;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: NumPower::array(self::CENTER)->toArray(),
            stdDev: 1.0
        );
    }

    #[Test]
    #[TestDox('Simulates a blob generator from dataset')]
    public function simulate() : void
    {
        $dataset = $this->generator->generate(100);

        $generator = Blob::simulate($dataset);

        self::assertInstanceOf(Blob::class, $generator);
        self::assertInstanceOf(Generator::class, $generator);
    }

    #[Test]
    #[TestDox('Returns center coordinates')]
    public function center() : void
    {
        self::assertEquals(
            NumPower::array(self::CENTER)->toArray(),
            $this->generator->center()
        );
    }

    #[Test]
    #[TestDox('Returns dimensions')]
    public function dimensions() : void
    {
        self::assertEquals(3, $this->generator->dimensions());
    }

    #[Test]
    #[TestDox('Generates an unlabeled dataset')]
    public function generate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        self::assertInstanceOf(Unlabeled::class, $dataset);
        self::assertInstanceOf(Dataset::class, $dataset);

        self::assertCount(self::DATASET_SIZE, $dataset);
    }
}
